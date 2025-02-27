import contextlib
from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure

__all__ = ["MongoConfig", "mongo_db"]

class MongoConfig:
    """
    Stores configuration details for connecting to a MongoDB database.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 27017,
        username: str = None,
        password: str = None,
        database_name: str = None,
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
        client = MongoClient(host=cfg.host,
                             port=cfg.port,
                             username=cfg.username,
                             password=cfg.password,
                             # ssl=cfg['mongo_ssl'],
                             # ssl_cert_reqs=ssl.CERT_NONE
                            )
        the_db = client[cfg.database_name]
    except ConnectionFailure as e:
        print(f"Failed to connect to MongoDB: {e}")
        raise

    try:
        yield the_db
    finally:
        client.close()


def create_index_if_not_exist(db, coll, index_name, field_name, is_unique):
    if index_name not in db[coll].index_information():
        print(f"Index on {field_name} does not exist. Creating it.")
        # Create the unique index on the `expr_has` field
        db[coll].create_index(
            [(field_name, ASCENDING)], name=index_name, unique=is_unique
        )
        print(f"Created index {index_name} successfully.")