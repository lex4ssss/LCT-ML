from datetime import datetime
from pathlib import Path


def build_examples(connection, gap_seconds, validation_start, test_start):
    if type(gap_seconds) is not int or not 0 < gap_seconds < 86400:
        raise ValueError('gap_seconds')
    if any(type(t) is not datetime or t.tzinfo is not None for t in (validation_start,test_start)):
        raise ValueError('split_timestamp')
    if validation_start >= test_start:
        raise ValueError('split_order')
    connection.execute('SET VARIABLE ml_gap_seconds = '+str(gap_seconds))
    for name,value in [('ml_validation_start',validation_start),('ml_test_start',test_start)]:
        connection.execute("SET VARIABLE "+name+" = TIMESTAMP '"+value.isoformat(sep=' ')+"'")
    query = Path(__file__).with_suffix('.sql').read_text()
    connection.execute('CREATE OR REPLACE TEMP VIEW training_examples AS '+query)
