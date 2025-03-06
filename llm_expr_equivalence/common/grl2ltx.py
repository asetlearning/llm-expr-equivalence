import re

import httpx

from llm_expr_equivalence.config.settings import get_settings


def remove_textcolor(text: str, q_param: str | None = None, replace: bool = False):
    q = "\\\\textcolor{" + (q_param if q_param is not None else "(.*?)") + "}{(.*?)}"
    d = re.search(q, text)
    if d is None:
        return None
    np = 0  # use stack to count parenthesis inside the textcolor
    pos_last = len(text)
    pos = d.end() + 1
    for c in text[d.end() + 1:]:
        if c == "{":
            np += 1
        elif c == "}" and np > 0:
            np -= 1
        elif c == "}" and np == 0:
            pos_last = pos
            break
        pos += 1

    if pos_last == len(text):
        raise Exception("Did not find the end of textcolor statement")
    if np != 0:
        raise Exception(f"Unbalanced number of parenthesis: {np}")

    inner_text = text[d.end() + 1:pos_last]
    if replace:
        result = text[:d.start()] + inner_text + text[pos_last + 1:]
    #        result = text.replace(d.group(0)+"{"+inner_text+"}", inner_text)
    else:
        result = inner_text
    return result


class ParserSyntaxException(Exception):
    """Exception with failed parser details"""

    def __init__(self, expr: str, response: dict) -> None:
        super().__init__()
        self.expr = expr
        self.message = str(dict)


class ParseGrlServer:
    def __init__(self):
        self.parser_client = httpx.Client(
            timeout=10.0,  # Set a timeout (optional)
            #            headers={"User-Agent": "MyApp/1.0"},  # Add default headers (optional)
        )

    def parse(self, grl: str) -> str:
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": "1",
                "method": "grl2latex",
                "params": [grl, None],
            }

            response = self.parser_client.post(
                url=get_settings().parser_url,  # URL from the settings
                json=payload,  # Pass the payload as JSON
            )
            response.raise_for_status()  # Raise an error on HTTP errors (4xx, 5xx)
            response_json = response.json()
            if "result" in response_json:
                if "textcolor" in response_json["result"]:
                    clean_result = remove_textcolor(response_json["result"])
                    print(f"Remove textcolor {response_json["result"]} -> {clean_result}")
                    return clean_result

                return response_json["result"]

            raise ParserSyntaxException(grl, response_json)

        except httpx.HTTPError as e:
            print(f"An error occurred while parsing {grl}: {e}")
            raise e

    def __del__(self):
        """
        Clean up the HTTPX client if needed.
        """
        self.parser_client.close()
