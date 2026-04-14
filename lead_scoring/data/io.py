import polars as pl
import pandera.polars as pa
from pandera.typing.polars import DataFrame

from lead_scoring.data.schema import RawDealsSchema


@pa.check_types
def read_lead_data(path: str) -> DataFrame[RawDealsSchema]:
    return DataFrame[RawDealsSchema](pl.read_csv(path))