from mathserver_client.models.mathserver_api.mathServerRequest import MathServerRequest
from mathserver_client.models.mathserver_api.solution import MathSolution, Expressions
from mathserver_client.models.mathserver_api.problem import MathProblem

from llm_expr_equivalence.common.mathserver import MathServer
from llm_expr_equivalence.db.mongo_tools import MongoConfig, mongo_db
from llm_expr_equivalence.common.grl2ltx import ParseGrlServer

from pymongo import ASCENDING, errors
import json
import hashlib

from pydantic import BaseModel
from enum import StrEnum


class AssistMethod(BaseModel):
    parameters: dict
    problemGraph: str
    req_timeout: int
    solution: MathSolution


class SourceType(StrEnum):
    student = "student"
    reference = "reference"
    answer = "answer"


class ExprPair:
    def __init__(self, e1: str, e2: str, src: SourceType):
        self.expr1: str = e1
        self.expr2: str = e2
        self.source: SourceType = src

    def __eq__(self, other):
        return (self.expr1 == other.expr1 and self.expr2 == other.expr2) or (
            self.expr1 == other.expr2 and self.expr2 == other.expr1
        )

    def __hash__(self):
        he1 = hash(self.expr1)
        he2 = hash(self.expr2)

        # TODO: check for better way to combine two hashes
        return hash(str(min(he1, he2)) + str(max(he1, he2)))

    def db_hash(self):
        # __hash__ is non-deterministic this gives persistent unique hash
        s = (
            f"{self.expr1}|#|{self.expr2}"
            if self.expr1 < self.expr2
            else f"{self.expr2}|#|{self.expr1}"
        )
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def __str__(self):
        return (
            f"({self.expr1}, {self.expr2}, src: {self.source}, hash: {self.db_hash()})"
        )


class ExtractExpressions:
    def __init__(self, request: MathServerRequest[AssistMethod]):
        self.solution = request.params.solution
        problem_graph_json = json.loads(request.params.problemGraph)
        self.problem = MathProblem.model_validate(problem_graph_json["the_problem"])
        self.ref_solutions = [
            MathSolution.model_validate(s["sol_structure"])
            for s in problem_graph_json["ref_solutions"]
        ]

        self.collected_expressions: set[ExprPair] = set()

    def add_exprs(self, e1: str, e2: str, src: SourceType):
        if e1 != e2:
            self.collected_expressions.add(ExprPair(e1, e2, src))

    def collect_solution_expressions(
        self, solution: MathSolution, src: SourceType
    ) -> int:
        expr_count = 0
        # collect rewrites
        for row in solution.rows:
            if len(row.elements) <= 1:
                continue
            for i in range(1, len(row.elements)):
                prev_expr = row.elements[i - 1].content
                next_expr = row.elements[i].content
                if isinstance(prev_expr, Expressions) and isinstance(
                    next_expr, Expressions
                ):
                    if (
                        len(prev_expr.expressions) != 1
                        or len(next_expr.expressions) != 1
                    ):
                        break
                    self.add_exprs(
                        prev_expr.expressions[0].expression.model_dump(),
                        next_expr.expressions[0].expression.model_dump(),
                        src,
                    )
                    expr_count += 1

        # collect answer expressions (all combinations)
        return expr_count

    @staticmethod
    def collect_answers(solution: MathSolution, result: dict) -> dict:
        # collect expression placeholders
        if solution.answers:
            for answer in solution.answers:
                placeholder_expressions = []
                for elt in answer.elements:
                    if elt.expression:
                        placeholder_expressions.append(elt.expression.model_dump())
                if placeholder_expressions:
                    result[answer.placeholder] = (
                        result.get(answer.placeholder, []) + placeholder_expressions
                    )

        return result

    def collect_all(self) -> int:
        total_count = self.collect_solution_expressions(
            self.solution, SourceType.student
        )
        for solution in self.ref_solutions:
            total_count += self.collect_solution_expressions(
                solution, SourceType.reference
            )

        # Add answers
        ref_ad = {}
        for solution in self.ref_solutions:
            ref_ad = self.collect_answers(solution, ref_ad)

        stud_ad = self.collect_answers(self.solution, {})

        # add answer placeholders to expressions
        for key, value in stud_ad.items():
            for stud_expr in value:
                for ref_expr in ref_ad[key]:
                    total_count += 1
                    # student expression is expr1, it matters for matching
                    self.add_exprs(stud_expr, ref_expr, SourceType.answer)

        return total_count


def collect_and_store():

    # collect expression pairs from the assists
    in_cfg = MongoConfig(
        #        host="localhost",
        username="",
        password="",
        #        'mongo_ssl': False,
        database_name="macs_data",
    )

    total_count = 0
    all_expressions_set: set[ExprPair] = set()
    with mongo_db(in_cfg) as db:
        for q in db.mathserver.find({"request.method": "assist"}):  # .limit(100):
            request = MathServerRequest[AssistMethod].model_validate(q["request"])
            extract_exprs = ExtractExpressions(request)
            cnt = extract_exprs.collect_all()
            total_count += cnt
            all_expressions_set.update(extract_exprs.collected_expressions)

    print(f"Collected: {total_count}.\nStored: {len(all_expressions_set)}")

    # store expressions in new db
    # collect expression pairs from the assists
    out_cfg = MongoConfig(
        #        host="localhost",
        username="",
        password="",
        #        'mongo_ssl': False,
        database_name="llm-expr-equivalence",
    )

    parser = ParseGrlServer()
    mathserver = MathServer()

    with mongo_db(out_cfg) as db:
        # Check if index already exists
        hash_index = (
            "expr_hash_1"  # Default index name for field 'expr_has' in ascending order
        )
        if hash_index not in db["expressions"].index_information():
            print(f"Unique index on 'expr_hash' does not exist. Creating it.")
            # Create the unique index on the `expr_has` field
            db["expressions"].create_index(
                [("exprs_hash", ASCENDING)], name=hash_index, unique=True
            )
            print("Created index on 'expr_hash' successfully.")

        for exprs in all_expressions_set:
            h = exprs.db_hash()
            expr_types = mathserver.get_types([exprs.expr1, exprs.expr2])
            if not expr_types or len(expr_types) != 2:
                raise Exception(f"Could not obtain types of {exprs.expr1}, {exprs.expr2}")

            are_of_type = mathserver.are_of_type([exprs.expr1, exprs.expr2], "Constant")
            if not are_of_type or len(are_of_type) != 2:
                raise Exception(f"Could not decide if expressions constant: {exprs.expr1}, {exprs.expr2}")

            try:
                result = db.expressions.insert_one(
                    {
                        "exprs_hash": exprs.db_hash(),
                        "expr1": {
                            "grl": exprs.expr1,
                            "ltx": parser.parse(exprs.expr1),
                            "type": expr_types[0],
                            "is_const": are_of_type[0],
                        },
                        "expr2": {
                            "grl": exprs.expr2,
                            "ltx": parser.parse(exprs.expr2),
                            "type": expr_types[1],
                            "is_const": are_of_type[1],
                        },
                        "source": exprs.source,
                    }
                )
                if result.inserted_id == 0:
                    print(f"Failed insert {exprs}")
            except errors.DuplicateKeyError as e:
                print(f"Skip existing: {exprs}: {e}")
