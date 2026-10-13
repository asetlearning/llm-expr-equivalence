import hashlib
import json
from enum import StrEnum

from mathserver_client.models.mathserver_api.mathServerRequest import MathServerRequest
from mathserver_client.models.mathserver_api.problem import MathProblem
from mathserver_client.models.mathserver_api.solution import Expressions, MathSolution
from pydantic import BaseModel
from pymongo import ASCENDING, errors
from tqdm import tqdm

from llm_expr_equivalence.common.grl2ltx import ParseGrlServer
from llm_expr_equivalence.common.mathserver import MathServer
from llm_expr_equivalence.config.settings import logger
from llm_expr_equivalence.db.mongo_tools import MongoConfig, mongo_db

MONGODB_PORT=27019

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
        for row in solution.rows or []:
            elements = row.elements or []
            if len(elements) <= 1:
                continue
            for i in range(1, len(elements)):
                prev_expr = elements[i - 1].content
                next_expr = elements[i].content
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
        ref_ad: dict = {}
        for solution in self.ref_solutions:
            ref_ad = self.collect_answers(solution, ref_ad)

        stud_ad = self.collect_answers(self.solution, {})

        # add answer placeholders to expressions
        for key, value in stud_ad.items():
            for stud_expr in value:
                try:
                    for ref_expr in ref_ad[key]:
                        total_count += 1
                        # student expression is expr1, it matters for matching
                        self.add_exprs(stud_expr, ref_expr, SourceType.answer)
                except KeyError:
                    logger.error(
                        f"No reference answer with key {key}: student {stud_ad}, ref {ref_ad}"
                    )

        return total_count


def add_exprs_to_db(
    db, exprs: ExprPair
):
    expr_types = [None, None]  # TODO: mathserver.get_types([exprs.expr1, exprs.expr2])
    if not expr_types or len(expr_types) != 2:
        raise Exception(f"Could not obtain types of {exprs.expr1}, {exprs.expr2}")

    are_of_type = [
        False,
        False,
    ]  # TODO: mathserver.are_of_type([exprs.expr1, exprs.expr2], "Constant")
    if not are_of_type or len(are_of_type) != 2:
        logger.error(
            f"Could not decide if expressions constant: {exprs.expr1}, {exprs.expr2}"
        )
        are_of_type = [False, False]

    try:
        result = db.expressions.insert_one(
            {
                "exprs_hash": exprs.db_hash(),
                "expr1": {
                    "grl": exprs.expr1,
                    "ltx": "undef",  # TODO: parser.parse(exprs.expr1),
                    "type": expr_types[0],
                    "is_const": are_of_type[0],
                },
                "expr2": {
                    "grl": exprs.expr2,
                    "ltx": "undef",  # TODO:  parser.parse(exprs.expr2),
                    "type": expr_types[1],
                    "is_const": are_of_type[1],
                },
                "source": exprs.source,
            }
        )
        if result.inserted_id == 0:
            logger.error(f"Failed insert {exprs}")
    except errors.DuplicateKeyError as e:
        logger.info(f"Skip existing: {exprs}: {e}")
    except Exception as e:
        logger.error(f"Failed insert {exprs}: {e}")


def label_exprs_in_db(  # noqa: C901
    db,
    exprs_record: dict,
    mathserver: MathServer,
    parser: ParseGrlServer
):
    expr_types: list[str | None] = [None, None]
    are_of_type: list[bool] = [False, False]
    expr1_ltx = None
    expr2_ltx = None

    expr1 = exprs_record["expr1"]["grl"]
    expr2 = exprs_record["expr2"]["grl"]

    if exprs_record["expr1"]["type"] is None or exprs_record["expr2"]["type"] is None:
        try:
            types = mathserver.get_types([expr1, expr2])
            if not types or len(types) != 2:
                logger.error(f"Could not obtain types of {expr1}, {expr2}")
            else:
                expr_types = list(types)
        except Exception as e:
            logger.error(f"Failed to get types of expressions: {expr1}, {expr2}: {e}")

        try:
            is_const = mathserver.are_of_type([expr1, expr2], "Constant")
            if not is_const or len(is_const) != 2:
                logger.error(
                    f"Could not decide if expressions constant: {expr1}, {expr2}"
                )
            else:
                are_of_type = is_const
        except Exception as e:
            logger.error(f"Failed to decide if expressions constant: {expr1}, {expr2}: {e}")


    if exprs_record["expr1"]["ltx"] == "undef" or exprs_record["expr1"]["ltx"] is None:
        try:
            expr1_ltx = parser.parse(expr1)
        except Exception as e:
            logger.error(f"Failed to parse expression: {expr1}: {e}")

    if exprs_record["expr2"]["ltx"] == "undef" or exprs_record["expr2"]["ltx"] is None:
        try:
            expr2_ltx = parser.parse(expr2)
        except Exception as e:
            logger.error(f"Failed to parse expression: {expr2}: {e}")

    # Update MongoDB record with new "type" and "is_const" values
    try:
        db.expressions.update_one(
            {"exprs_hash": exprs_record["exprs_hash"]},
            {
                "$set": {
                    "expr1.type": expr_types[0],
                    "expr1.is_const": are_of_type[0],
                    "expr1.ltx":expr1_ltx,
                    "expr2.type": expr_types[1],
                    "expr2.is_const": are_of_type[1],
                    "expr2.ltx":expr2_ltx,
            }
        },
        )
    except Exception as e:
        logger.error(f"Failed to update record: {exprs_record}: {e}")


def collect_and_store_save(input_db: str | None = None, output_db: str | None = None):

    # collect expression pairs from the assists
    in_cfg = MongoConfig(
        #        host="localhost",
        port=MONGODB_PORT,
        username="",
        password="",
        #        'mongo_ssl': False,
        database_name=input_db if input_db is not None else "grd",
    )

    # store expressions in new db
    # collect expression pairs from the assists
    out_cfg = MongoConfig(  # noqa: F841 - used by commented-out block below
        #        host="localhost",
        port=MONGODB_PORT,
        username="",
        password="",
        #        'mongo_ssl': False,
        database_name=output_db if output_db else "llm-expr-equivalence",
    )

    # TODO: Need to extract expressions in two passes.
    #  It looks like number of unique expressions is significantly les then the actual expressions
    #  First extract `unique` expressions, then sent to parser and  isOfType
    #  Creating set in memory is probably was the right thing.
    total_count = 0
    all_expressions_set: set[ExprPair] = set()
    with mongo_db(in_cfg) as db_in:
        for q in tqdm(
            db_in.mathserver.find({"request.method": "assist"}),  # .limit(100),
            desc="Assists",
        ):
            try:
                request = MathServerRequest[AssistMethod].model_validate(q["request"])
            except Exception as e:
                logger.error(f"Failed to parse request: {q['request']}: {str(e)}")
                continue
            extract_exprs = ExtractExpressions(request)
            cnt = extract_exprs.collect_all()
            total_count += cnt
            all_expressions_set.update(extract_exprs.collected_expressions)

    print(f"Collected: {total_count}.\nStored: {len(all_expressions_set)}")

    # parser = ParseGrlServer()
    # mathserver = MathServer()
    #
    # with mongo_db(out_cfg) as db_out:
    #     # Check if index already exists
    #     hash_index = (
    #         "expr_hash_1"  # Default index name for field 'expr_has' in ascending order
    #     )
    #     if hash_index not in db_out["expressions"].index_information():
    #         logger.info("Unique index on 'expr_hash' does not exist. Creating it.")
    #         # Create the unique index on the `expr_has` field
    #         db_out["expressions"].create_index(
    #             [("exprs_hash", ASCENDING)], name=hash_index, unique=True
    #         )
    #         logger.info("Created index on 'expr_hash' successfully.")
    #
    #     total_count = 0
    #     added_count = 0
    #     with mongo_db(in_cfg) as db_in:
    #         for q in tqdm(db_in.mathserver.find({"request.method": "assist"}), #.limit(100),
    #                       desc="Processing expressions"):
    #             request = MathServerRequest[AssistMethod].model_validate(q["request"])
    #             extract_exprs = ExtractExpressions(request)
    #             cnt = extract_exprs.collect_all()
    #             total_count += cnt
    #             for expr in extract_exprs.collected_expressions:
    #                 add_exprs_to_db(db_out, expr, mathserver, parser)
    #                 added_count += 1
    #
    # logger.info(f"Collected: {total_count}. Stored: {added_count}")


def collect_and_store(input_db: str | None = None, output_db: str | None = None):

    # collect expression pairs from the assists
    in_cfg = MongoConfig(
        #        host="localhost",
        port=MONGODB_PORT,
        username="",
        password="",
        #        'mongo_ssl': False,
        database_name=input_db if input_db is not None else "grd",
    )

    # store expressions in new db
    # collect expression pairs from the assists
    out_cfg = MongoConfig(
        #        host="localhost",
        port=MONGODB_PORT,
        username="",
        password="",
        #        'mongo_ssl': False,
        database_name=output_db if output_db else "llm-expr-equivalence",
    )

    with mongo_db(out_cfg) as db_out:
        # Check if index already exists
        hash_index = (
            "expr_hash_1"  # Default index name for field 'expr_has' in ascending order
        )
        if hash_index not in db_out["expressions"].index_information():
            logger.info("Unique index on 'expr_hash' does not exist. Creating it.")
            # Create the unique index on the `expr_has` field
            db_out["expressions"].create_index(
                [("exprs_hash", ASCENDING)], name=hash_index, unique=True
            )
            logger.info("Created index on 'expr_hash' successfully.")

    # TODO: Need to extract expressions in batches using two-pass.
    #  It looks like number of unique expressions is significantly les then the actual expressions
    #  First extract `unique` expressions in a btach then sent to parser and  isOfType and add to mongo
    batch_size = 1000000
    n_batches = 58
    total_count = 0
    total_added = 0

    with mongo_db(in_cfg) as db_in:
        has_records = False
        for batch_number in range(n_batches):
            all_expressions_set: set[ExprPair] = set()
            for q in tqdm(
                db_in.mathserver.find({"request.method": "assist"})
                .skip(batch_number * batch_size)
                .limit(batch_size),
                desc=f"Batch: {batch_number} Assists: ",
            ):
                has_records = True
                try:
                    request = MathServerRequest[AssistMethod].model_validate(
                        q["request"]
                    )
                    extract_exprs = ExtractExpressions(request)
                    cnt = extract_exprs.collect_all()
                    total_count += cnt
                    all_expressions_set.update(extract_exprs.collected_expressions)

                except Exception as e:
                    logger.error(f"Failed to process request: {q['request']}: {str(e)}")
                    continue

            total_added += len(all_expressions_set)
            print(f"Batch {batch_number}: Collected={total_count}. Unique={total_added}")

            # Store batch in mongo. This may reduce some duplicates
            with mongo_db(out_cfg) as db_out:
                for expr in tqdm(all_expressions_set, desc=f"Batch {n_batches}. Store in mongo: "):
                    add_exprs_to_db(db_out, expr)

            if not has_records:
                break

    print(f"Total: Collected={total_count}. Unique={total_added}")


def label_expressions(input_db: str | None = None):
    in_cfg = MongoConfig(
        #        host="localhost",
        port=MONGODB_PORT,
        username="",
        password="",
        #        'mongo_ssl': False,
        database_name=input_db if input_db else "llm-expr-equivalence",
    )

    parser = ParseGrlServer()
    mathserver = MathServer()

    with mongo_db(in_cfg) as db_in:
        query = {
        "$or": [
            {"expr1.type": None},
            {"expr2.type": None}
        ]
        }
        for q in tqdm(
            db_in.expressions.find(query, batch_size=10, no_cursor_timeout=True), #.limit(100), # .skip(190000),
            desc="Label expressions: ",
        ):
#            print(" >>>>>>>>>>>>> ", q)
            try:
                label_exprs_in_db(db_in, q, mathserver, parser)
            except Exception as e:
                logger.error(f"Failed to label expression: {q}: {str(e)}")
                continue
