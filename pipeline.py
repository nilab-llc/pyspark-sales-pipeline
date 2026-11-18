"""Stage 1: deterministic synthetic sales, DataFrame/SQL parity checks."""
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from pyspark.sql import SparkSession, Window, functions as F

ROOT = Path(__file__).resolve().parent


def main():
    spark = (SparkSession.builder.master('local[2]').appName('sales-stage-1')
             .config('spark.sql.session.timeZone', 'UTC')
             .config('spark.sql.ansi.enabled', 'true')
             .config('spark.sql.shuffle.partitions', '2')
             .config('spark.ui.enabled', 'false').getOrCreate())
    spark.sparkContext.setLogLevel('ERROR')
    output = ROOT / 'results'
    output.mkdir(exist_ok=True)
    try:
        customers = spark.createDataFrame([
            (' C001 ', ' Alice '), ('C002', ' Bob'), ('C003', 'Carol '),
            ('C004', 'No orders')], 'customer_id string, customer_name string')
        products = spark.createDataFrame([
            (' P01 ', ' Keyboard ', '100.00'), ('P02', ' Mouse ', '25.50'),
            ('P03', 'Cable', '10.00')],
            'product_id string, product_name string, unit_price string')
        rows = [
            ('O01',' C001 ','P01','2026-01-05','2'),
            ('O02','C002',' P02 ','2026-01-12','4'),
            ('O03','C001','P03','2026-02-01','3'),
            ('O04','C003','P01','2026-02-14','1'),
            ('O05','C002','P02','2026-02-20','2'),
            ('O06','C003','P03','2026-03-03','5'),
            ('O07','C002','P03','2026-03-15','5'),
            ('O08','C001','P02','2026-03-31','1'),
            ('O09','C001','P03','2026-02-28','1'),
            ('O09','C001','P03','2026-02-28','1'),  # exact duplicate
            ('O10',None,'P01','2026-01-01','1'),
            ('O11','C001',None,'2026-01-01','1'),
            ('O12','C001','P01','bad-date','1'),
            ('O13','C001','P01','2026-01-01','bad'),
            ('O14','C001','P01','2026-01-01','0'),
            ('O15','C001','P01','2026-01-01','-1'),
            ('O16','C999','P01','2026-01-01','1'),
            ('O17','C001','P99','2026-01-01','1'),
            ('O18','C001','P01',None,'1'),
            ('O19','C001','P01','2026-01-01',None),
            ('O20','   ','P01','2026-01-01','1')]
        orders = spark.createDataFrame(rows,
            'order_id string, customer_id string, product_id string, order_date string, quantity string')
        # Retain raw fixtures so all input data can be inspected without Spark.
        fixtures = {'orders': [r.asDict() for r in orders.collect()],
                    'customers': [r.asDict() for r in customers.collect()],
                    'products': [r.asDict() for r in products.collect()]}
        (output / 'inputs.json').write_text(json.dumps(fixtures, indent=2) + '\n')
        customers = customers.select(F.trim('customer_id').alias('customer_id'),
                                     F.trim('customer_name').alias('customer_name'))
        products = products.select(F.trim('product_id').alias('product_id'),
            F.trim('product_name').alias('product_name'),
            F.expr('try_cast(unit_price as decimal(12,2))').alias('unit_price'))
        # Dimension uniqueness is required to prevent join multiplication.
        for frame, key in [(customers, 'customer_id'), (products, 'product_id')]:
            assert frame.filter(F.col(key).isNull() | (F.col(key) == '')).count() == 0
            assert frame.groupBy(key).count().filter('count > 1').count() == 0
        clean = orders.select(
            *[F.trim(c).alias(c) for c in ['order_id','customer_id','product_id']],
            F.expr('try_cast(order_date as date)').alias('order_date'),
            F.expr('try_cast(quantity as int)').alias('quantity')).distinct()
        valid = clean.dropna().filter(
            (F.col('order_id') != '') & (F.col('customer_id') != '') &
            (F.col('product_id') != '') & (F.col('quantity') > 0))
        joined = (valid.join(customers, 'customer_id').join(products, 'product_id')
                  .filter(F.col('unit_price').isNotNull() & (F.col('unit_price') >= 0))
                  .withColumn('amount', (F.col('quantity') * F.col('unit_price')).cast('decimal(18,2)')))
        monthly_df = joined.groupBy(F.date_format('order_date','yyyy-MM').alias('month')).agg(
            F.sum('amount').alias('total_sales'), F.count('*').alias('order_count')).orderBy('month')
        totals = joined.groupBy('customer_id','customer_name').agg(F.sum('amount').alias('total_sales'))
        ranking_df = totals.withColumn('sales_rank', F.dense_rank().over(
            Window.orderBy(F.col('total_sales').desc()))).orderBy('sales_rank','customer_id')
        # SQL independently repeats normalization, filtering and joins from RAW views.
        orders.createOrReplaceTempView('raw_orders')
        spark.createDataFrame(fixtures['customers'], customers.schema).createOrReplaceTempView('raw_customers')
        spark.createDataFrame(fixtures['products'], 'product_id string, product_name string, unit_price string').createOrReplaceTempView('raw_products')
        spark.sql('''CREATE OR REPLACE TEMP VIEW sql_sales AS
            WITH normalized AS (
              SELECT DISTINCT trim(order_id) order_id, trim(customer_id) customer_id,
                trim(product_id) product_id, try_cast(order_date AS DATE) order_date,
                try_cast(quantity AS INT) quantity FROM raw_orders
            ), c AS (SELECT trim(customer_id) customer_id, trim(customer_name) customer_name FROM raw_customers),
            p AS (SELECT trim(product_id) product_id, trim(product_name) product_name,
                try_cast(unit_price AS DECIMAL(12,2)) unit_price FROM raw_products)
            SELECT o.*, c.customer_name, p.product_name, p.unit_price,
                cast(o.quantity * p.unit_price AS DECIMAL(18,2)) amount
            FROM normalized o JOIN c ON o.customer_id=c.customer_id
              JOIN p ON o.product_id=p.product_id
            WHERE o.order_id IS NOT NULL AND o.order_id <> ''
              AND o.customer_id <> '' AND o.product_id <> ''
              AND o.order_date IS NOT NULL AND o.quantity > 0
              AND p.unit_price >= 0''')
        monthly_sql = spark.sql('''SELECT date_format(order_date,'yyyy-MM') month,
            sum(amount) total_sales, count(*) order_count FROM sql_sales
            GROUP BY date_format(order_date,'yyyy-MM') ORDER BY month''')
        ranking_sql = spark.sql('''WITH totals AS (
            SELECT customer_id, customer_name, sum(amount) total_sales FROM sql_sales
            GROUP BY customer_id, customer_name)
            SELECT *, dense_rank() OVER (ORDER BY total_sales DESC) sales_rank
            FROM totals ORDER BY sales_rank, customer_id''')
        checks = {}
        for name, left, right in [('monthly_sales',monthly_df,monthly_sql),
                                  ('customer_ranking',ranking_df,ranking_sql)]:
            assert left.schema == right.schema, f'{name}: schema mismatch'
            assert left.exceptAll(right).count() == 0, f'{name}: DF-only rows'
            assert right.exceptAll(left).count() == 0, f'{name}: SQL-only rows'
            a, b = left.collect(), right.collect()
            assert a == b, f'{name}: order mismatch'
            for suffix, data in [('dataframe', a), ('sql', b)]:
                (output / f'{name}_{suffix}.json').write_text(
                    json.dumps([r.asDict() for r in data], default=str, indent=2) + '\n')
            checks[name] = {'schema_equal': True, 'multiset_equal_both_directions': True,
                            'ordered_rows_equal': True, 'rows': len(a)}
        assert [(r.month, str(r.total_sales), r.order_count) for r in monthly_df.collect()] == [
            ('2026-01','302.00',2), ('2026-02','191.00',4), ('2026-03','125.50',3)]
        assert [(r.customer_id, str(r.total_sales), r.sales_rank) for r in ranking_df.collect()] == [
            ('C001','265.50',1), ('C002','203.00',2), ('C003','150.00',3)]
        # Tie case: rank depends only on sales, display ties by customer_id.
        tie = spark.createDataFrame([('A',10),('B',10),('C',5)], 'customer_id string, total_sales int')
        tie.createOrReplaceTempView('tie_totals')
        tie_df = tie.withColumn('sales_rank', F.dense_rank().over(Window.orderBy(F.desc('total_sales')))).orderBy('sales_rank','customer_id')
        tie_sql = spark.sql('SELECT *, dense_rank() OVER (ORDER BY total_sales DESC) sales_rank FROM tie_totals ORDER BY sales_rank, customer_id')
        assert tie_df.collect() == tie_sql.collect()
        assert [r.sales_rank for r in tie_df.collect()] == [1,1,2]
        counts = {'raw_orders': orders.count(), 'distinct_normalized_orders': clean.count(),
                  'valid_before_join': valid.count(), 'joined_orders': joined.count()}
        assert counts == dict(raw_orders=21, distinct_normalized_orders=20, valid_before_join=11, joined_orders=9)
        report = {'status':'PASS', 'executed_at_utc':datetime.now(timezone.utc).isoformat(),
                  'python':platform.python_version(), 'spark':spark.version,
                  'java':spark._jvm.java.lang.System.getProperty('java.version'),
                  'master':spark.sparkContext.master, 'counts':counts, 'checks':checks,
                  'expected_results_assertions':'PASS', 'tie_ranking_assertion':'PASS'}
        (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
        monthly_df.show(truncate=False)
        ranking_df.show(truncate=False)
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
