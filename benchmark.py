"""Cloud-local skewed join benchmark; metrics come exclusively from Spark events."""
import argparse
import json
import statistics
import time
from pathlib import Path
from pyspark.sql import SparkSession, functions as F

ROOT = Path(__file__).resolve().parent


def inputs(spark, n):
    # 90% of orders belong to customer 0; remaining orders span 10,000 customers.
    orders = spark.range(n, numPartitions=8).select(
        F.col('id').alias('order_id'),
        F.when(F.col('id') % 10 != 0, F.lit(0)).otherwise(
            (F.col('id') / 10).cast('long') % 10000 + 1).alias('customer_id'),
        (F.col('id') % 100 + 1).alias('amount'))
    customers = spark.range(10001, numPartitions=2).select(
        F.col('id').alias('customer_id'), (F.col('id') % 5).alias('segment'))
    return orders, customers


def query(orders, customers, method):
    hint = 'merge' if method == 'sort_merge' else 'broadcast'
    return orders.hint('merge').join(customers.hint(hint), 'customer_id') if method == 'sort_merge' else orders.join(customers.hint(hint), 'customer_id')


def sink(joined):
    # Consume both tables and all joined rows; collect only five output groups.
    return joined.groupBy('segment').agg(F.count('*').alias('orders'),
        F.sum('amount').alias('sales'), F.sum('order_id').alias('order_id_sum')).orderBy('segment')


def percentile(values, fraction):
    values = sorted(values)
    pos = (len(values) - 1) * fraction
    lo = int(pos)
    return values[lo] + (values[min(lo + 1, len(values)-1)] - values[lo]) * (pos-lo)


def parse_events(path, runs):
    stages, tasks, groups, executions = {}, {}, {}, {}
    def nodes(node):
        yield node
        for child in node.get('children', []):
            yield from nodes(child)
    for file in sorted(path.rglob('*')):
        if not file.is_file() or file.name.startswith('.') or file.name.endswith('.crc'):
            continue
        with file.open() as stream:
            for line in stream:
                e = json.loads(line)
                kind = e['Event']
                if kind == 'SparkListenerJobStart':
                    group = (e.get('Properties') or {}).get('spark.jobGroup.id')
                    if group:
                        groups.setdefault(group, set()).update(e['Stage IDs'])
                elif kind.endswith('SparkListenerSQLExecutionStart'):
                    executions[str(e['executionId'])] = e.get('sparkPlanInfo', {})
                elif kind == 'SparkListenerStageCompleted':
                    info = e['Stage Info']
                    stages[info['Stage ID']] = info
                elif kind == 'SparkListenerTaskEnd':
                    if e.get('Task End Reason', {}).get('Reason') == 'Success':
                        tasks.setdefault(e['Stage ID'], []).append(e)
    for run in runs:
        group = run['group']
        stage_ids = sorted(groups.get(group, []))
        # SQL join output-row accumulator uniquely connects physical join node to stage.
        join_acc = set()
        for plan in executions.values():
            for node in nodes(plan):
                if run['actual_join'] in node.get('nodeName', ''):
                    join_acc.update(m['accumulatorId'] for m in node.get('metrics', []) if m['name'] == 'number of output rows')
        join_stages = [sid for sid in stage_ids if any(
            a['ID'] in join_acc for a in stages.get(sid, {}).get('Accumulables', []))]
        run['job_stage_ids'] = stage_ids
        run['join_stage_ids'] = join_stages
        run['join_stages'] = []
        for sid in join_stages:
            ts = tasks.get(sid, [])
            durations = [t['Task Info']['Finish Time'] - t['Task Info']['Launch Time'] for t in ts]
            read = sum(t['Task Metrics'].get('Shuffle Read Metrics', {}).get('Remote Bytes Read', 0) +
                       t['Task Metrics'].get('Shuffle Read Metrics', {}).get('Local Bytes Read', 0) for t in ts)
            write = sum(t['Task Metrics'].get('Shuffle Write Metrics', {}).get('Shuffle Bytes Written', 0) for t in ts)
            median = statistics.median(durations) if durations else None
            run['join_stages'].append({'stage_id':sid, 'tasks':len(ts),
                'shuffle_read_bytes':read, 'shuffle_write_bytes':write,
                'task_duration_ms':durations, 'task_median_ms':median,
                'task_p95_ms':percentile(durations, .95) if durations else None,
                'task_max_ms':max(durations) if durations else None,
                'max_to_median':max(durations)/median if median else None})
        if not join_stages:
            run['join_metrics_unavailable_reason'] = 'No completed stage exposed the join output-row accumulator; no estimated metrics substituted.'
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, default=3000000)
    parser.add_argument('--small-only', action='store_true',
                        help='Verify 2000 full joined rows once; skip all performance runs')
    args = parser.parse_args()
    out = ROOT / ('review/small' if args.small_only else 'performance')
    out.mkdir(parents=True, exist_ok=True)
    events = out / 'events'
    events.mkdir(exist_ok=True)
    spark = (SparkSession.builder.master('local[2]').appName('skew-join-benchmark')
        .config('spark.ui.enabled','false').config('spark.sql.adaptive.enabled','false')
        .config('spark.sql.autoBroadcastJoinThreshold','-1')
        .config('spark.sql.shuffle.partitions','8')
        .config('spark.sql.join.preferSortMergeJoin','true')
        .config('spark.eventLog.enabled','true').config('spark.eventLog.compress','false')
        .config('spark.eventLog.dir',events.as_uri()).getOrCreate())
    spark.sparkContext.setLogLevel('ERROR')
    runs = []
    try:
        small = inputs(spark, 2000)
        left = query(*small, 'sort_merge').orderBy('order_id')
        right = query(*small, 'broadcast').orderBy('order_id')
        assert left.schema == right.schema
        assert left.collect() == right.collect()
        assert left.count() == 2000
        if args.small_only:
            actual = {}
            for name, frame, operator in [('sort_merge', left, 'SortMergeJoin'),
                                           ('broadcast', right, 'BroadcastHashJoin')]:
                plan = frame._jdf.queryExecution().executedPlan().toString()
                assert operator in plan, plan
                actual[name] = operator
                (out / f'{name}_plan.txt').write_text(frame._jdf.queryExecution().toString())
            report = {'status': 'PASS', 'orders': 2000, 'customers': 10001,
                      'full_output_schema_values_order_equal': True,
                      'actual_join': actual, 'aqe_enabled': spark.conf.get('spark.sql.adaptive.enabled'),
                      'spark': spark.version, 'java': spark._jvm.java.lang.System.getProperty('java.version'),
                      'master': spark.sparkContext.master,
                      'performance_runs': 0, 'large_dataset_executed': False}
            (out / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(report, indent=2), flush=True)
            return
        orders, customers = inputs(spark, args.rows)
        orders.cache(); customers.cache()
        assert orders.count() == args.rows
        assert customers.count() == 10001
        # One unmeasured warmup for each method, using the same cached inputs.
        for method in ['sort_merge','broadcast']:
            sink(query(orders, customers, method)).collect()
        expected = None
        for repeat in range(1,4):
            # Alternate order to reduce systematic order effects.
            methods = ['sort_merge','broadcast'] if repeat % 2 else ['broadcast','sort_merge']
            for method in methods:
                group = f'{method}_{repeat}'
                spark.sparkContext.setJobGroup(group,group)
                result = sink(query(orders, customers, method))
                start = time.perf_counter()
                data = result.collect()
                elapsed = time.perf_counter()-start
                if expected is None:
                    expected = data
                assert data == expected
                plan = result._jdf.queryExecution().executedPlan().toString()
                actual = 'SortMergeJoin' if 'SortMergeJoin' in plan else 'BroadcastHashJoin' if 'BroadcastHashJoin' in plan else 'UNKNOWN'
                assert actual == ('SortMergeJoin' if method == 'sort_merge' else 'BroadcastHashJoin'), plan
                (out / f'{group}_plan.txt').write_text(result._jdf.queryExecution().toString())
                runs.append({'method':method,'repeat':repeat,'group':group,'seconds':elapsed,'actual_join':actual})
                print(json.dumps(runs[-1]), flush=True)
        config = {k:spark.conf.get(k) for k in ['spark.sql.adaptive.enabled',
            'spark.sql.adaptive.skewJoin.enabled','spark.sql.autoBroadcastJoinThreshold',
            'spark.sql.shuffle.partitions','spark.sql.join.preferSortMergeJoin']}
        environment = {'spark':spark.version,'java':spark._jvm.java.lang.System.getProperty('java.version'),
                       'master':spark.sparkContext.master,'application_id':spark.sparkContext.applicationId}
        (out / 'output.json').write_text(json.dumps([r.asDict() for r in expected],indent=2)+'\n')
    finally:
        spark.stop()
    # Restrict parsing to this application; keep historical raw event logs untouched.
    app_paths = list(events.glob('*'+environment['application_id']+'*'))
    assert len(app_paths) == 1, app_paths
    parse_events(app_paths[0],runs)
    report = {'rows':args.rows,'customers':10001,'hot_customer_rows':args.rows-((args.rows+9)//10),
        'small_full_output_equality':'PASS (2000 rows, ordered full rows and schema)',
        'large_aggregate_equality':'PASS for all 6 runs', 'config':config,'environment':environment,
        'timing_scope':'collect() wall clock: planning, join, aggregate, order, result transfer; excludes fixture generation, cache materialization and warmups',
        'runs':runs,'median_seconds':{m:statistics.median(r['seconds'] for r in runs if r['method']==m) for m in ['sort_merge','broadcast']}}
    (out / 'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
