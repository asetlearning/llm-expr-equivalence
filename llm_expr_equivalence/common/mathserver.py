from httpx import URL, Client, Timeout

from llm_expr_equivalence.config.settings import get_settings, logger
from llm_expr_equivalence.schemas.common import Trichotomy


class MathServer:
    def __init__(self):
        """
        Init MathServer client.
        """
        timeout = Timeout(30.0, connect=1.0)

        self.url = URL(get_settings().mathserver_url)
        self.client = Client(
            base_url=self.url,
            http2=True,
            timeout=timeout,
        )

    def __del__(self):
        self.client.close()

    def post(self, request: dict) -> dict:
        """
        Send arbitrary request to mathsever and returns its response in json
        """
        response = self.client.post(self.url, json=request)
        response.raise_for_status()
        logger.info(
            f"Mathserver invoke: request = {request}, response = {response.json()}"
        )
        return response.json()

    def get_is_equivalent(self, lhs: str, rhs: str) -> Trichotomy:
        """
        Call  MathServer to check equivalence of two expressions
        TODO: move to mathserver_client calls
        """
        request = {
            "id": "1",
            "jsonrpc": "2.0",
            "method": "is_equivalent",
            "params": {"lhs": lhs, "rhs": rhs},
        }
        response = self.post(request)
        return response["result"]

    def get_does_match_reference(self, stud: str, ref: str) -> bool:
        """
        Call  MathServer to check if students expression matches reference given matching algorithm
        TODO: move to mathserver_client calls
        """
        request = {
            "id": "1",
            "jsonrpc": "2.0",
            "method": "does_match_primary",
            "params": {
                "primary_expr": [{"expression": ref}],
                "actual_expr": {"expression": stud},
                "matchType": "calculus",
                "request_feedback": False,
            },
        }
        response = self.post(request)
        if "status" in response["result"] and response["result"]["status"] == "success":
            return True
        else:
            return False

    def get_types(self, exprs: list[str]) -> list[str] | None:
        """
        Call  MathServer to get expression types
        """
        request = {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "get_expression_type",
            "params": {"expr_array": exprs},
        }
        response = self.post(request)
        if "result" in response:
            return response["result"]
        else:
            return None

    def are_of_type(self, exprs: list[str], t: str) -> list[bool] | None:
        """
        Call  MathServer to get expression types
        """
        request = {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "are_of_types",
            "params": {"expressions": exprs, "types": [t]},
        }
        response = self.post(request)
        if "result" in response:
            return response["result"]
        else:
            return None
