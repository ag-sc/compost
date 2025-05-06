import dspy


class FewShotFixedSignature(dspy.Signature):
    shot1_question = dspy.InputField(desc="first example question about something")
    shot1_sparql_query = dspy.InputField(desc="first example sparql query for DBpedia")
    shot2_question = dspy.InputField(desc="second example question about something")
    shot2_sparql_query = dspy.InputField(desc="second example sparql query for DBpedia")
    question = dspy.InputField(desc="question about something")
    sparql_query = dspy.OutputField(desc="sparql query for DBpedia")

class FewShotVariableFixedSignature(dspy.Signature):
    shots: list[tuple[str, str]] = dspy.InputField(desc="example question sparql query pairs")
    question: str = dspy.InputField(desc="question about something")
    sparql_query: str = dspy.OutputField(desc="sparql query for DBpedia")


class FewShotPipeline(dspy.Module):
    def __init__(self, predictor_module=None):
        super().__init__()
        self.signature = PlainSignature

        if isinstance(predictor_module, str) and predictor_module.lower() == "chainofthought":
            self.predictor = dspy.ChainOfThought(self.signature)
        elif isinstance(predictor_module, str) and predictor_module.lower() == "programofthought":
            self.predictor = dspy.ProgramOfThought(self.signature)
        else:
            self.predictor = dspy.Predict(self.signature)

    def forward(self, question):
        result = self.predictor(question=question)
        return dspy.Prediction(
            sparql_query=result.sparql_query if hasattr(result, 'sparql_query') else None,
            reasoning=result.reasoning if hasattr(result, 'reasoning') else None
        )


class FewShotFixedPipeline(dspy.Module):
    def __init__(self, predictor_module=None):
        super().__init__()
        self.signature = FewShotFixedSignature

        if isinstance(predictor_module, str) and predictor_module.lower() == "chainofthought":
            self.predictor = dspy.ChainOfThought(self.signature)
        elif isinstance(predictor_module, str) and predictor_module.lower() == "programofthought":
            self.predictor = dspy.ProgramOfThought(self.signature)
        else:
            self.predictor = dspy.Predict(self.signature)

    def forward(self,
                question,
                shot1_question,
                shot1_sparql_query,
                shot2_question,
                shot2_sparql_query,
    ):
        result = self.predictor(question=question, shot1_question=shot1_question, shot1_sparql_query=shot1_sparql_query, shot2_question=shot2_question, shot2_sparql_query=shot2_sparql_query)
        return dspy.Prediction(
            sparql_query=result.sparql_query if hasattr(result, 'sparql_query') else None,
            reasoning=result.reasoning if hasattr(result, 'reasoning') else None
        )

class FewShotVariableFixedPipeline(dspy.Module):
    def __init__(self, predictor_module=None):
        super().__init__()
        self.signature = FewShotVariableFixedSignature

        if isinstance(predictor_module, str) and predictor_module.lower() == "chainofthought":
            self.predictor = dspy.ChainOfThought(self.signature)
        elif isinstance(predictor_module, str) and predictor_module.lower() == "programofthought":
            self.predictor = dspy.ProgramOfThought(self.signature)
        else:
            self.predictor = dspy.Predict(self.signature)

    def forward(self,
                question,
                shots
    ):
        result = self.predictor(question=question, shots=shots)
        return dspy.Prediction(
            sparql_query=result.sparql_query if hasattr(result, 'sparql_query') else None,
            reasoning=result.reasoning if hasattr(result, 'reasoning') else None
        )


class PlainSignature(dspy.Signature):
    question = dspy.InputField(desc="question about something")
    sparql_query = dspy.OutputField(desc="sparql query for DBpedia")


class ZeroShotPipeline(dspy.Module):
    def __init__(self, predictor_module=None):
        super().__init__()

        self.signature = PlainSignature

        if isinstance(predictor_module, str) and predictor_module.lower() == "chainofthought":
            self.predictor = dspy.ChainOfThought(self.signature)
        elif isinstance(predictor_module, str) and predictor_module.lower() == "programofthought":
            self.predictor = dspy.ProgramOfThought(self.signature)
        else:
            self.predictor = dspy.Predict(self.signature)

    def forward(self, question):
        result = self.predictor(question=question)
        return dspy.Prediction(
            sparql_query=result.sparql_query if hasattr(result, 'sparql_query') else None,
            reasoning=result.reasoning if hasattr(result, 'reasoning') else None
        )
