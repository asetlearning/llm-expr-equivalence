import contextlib
from typing import Optional

from pymongo import ASCENDING, MongoClient
from pymongo.errors import ConnectionFailure

from llm_expr_equivalence.config.settings import logger

__all__ = ["MongoConfig", "mongo_db"]

class MongoConfig:
    """
    Stores configuration details for connecting to a MongoDB database.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 27017,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database_name: Optional[str] = None,
    ):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database_name = database_name


@contextlib.contextmanager
def mongo_db(cfg: MongoConfig):
    """
       Connect to mongodb.
    """
    try:
        client: MongoClient = MongoClient(host=cfg.host,
                             port=cfg.port,
                             username=cfg.username,
                             password=cfg.password,
                             # ssl=cfg['mongo_ssl'],
                             # ssl_cert_reqs=ssl.CERT_NONE
                            )
        the_db = client.get_database(cfg.database_name)
    except ConnectionFailure as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise

    try:
        yield the_db
    finally:
        client.close()


def create_index_if_not_exist(db, coll, index_name, field_name, is_unique):
    if index_name not in db[coll].index_information():
        logger.info(f"Index on {field_name} does not exist. Creating it.")
        # Create the unique index on the `expr_has` field
        db[coll].create_index(
            [(field_name, ASCENDING)], name=index_name, unique=is_unique
        )
        logger.info(f"Created index {index_name} successfully.")
