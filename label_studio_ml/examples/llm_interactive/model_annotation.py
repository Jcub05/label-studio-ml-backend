import logging
import json
import os

from bs4 import BeautifulSoup
from typing import List, Dict, Optional

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk.label_interface.objects import PredictionValue

from model import gpt

logger = logging.getLogger(__name__)


class AutoAnnotationAICall(LabelStudioMLBase):
    OPENAI_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o')
    FEW_SHOT_PATH = os.getenv('FEW_SHOT_PATH', 'few_shot_examples.txt')

    def setup(self):
        if self.FEW_SHOT_PATH and os.path.isfile(self.FEW_SHOT_PATH):
            with open(self.FEW_SHOT_PATH, 'r', encoding='utf-8') as f:
                self.few_shot_examples = f.read()
        else:
            self.few_shot_examples = ''
            logger.warning('FEW_SHOT_PATH not set or file not found — few-shot examples will be empty')

    def _strip_html(self, html: str) -> str:
        return BeautifulSoup(html, 'html.parser').get_text()

    def _get_task_data(self, task: Dict):
        task_data = task['data'].get('html', '')
        task_source = task['data'].get('source', '')
        task_year = task['data'].get('year', '')
        task_data_nohtml = self._strip_html(task_data)
        return (task_data_nohtml, task_source, task_year)

    def _build_system_prompt(self) -> str:
        schema = """You are an expert legal annotator specializing in Indonesian employment law cases.

Given a court document, read it carefully and classify it using EXACTLY the fields and valid options below.
Output ONLY a valid JSON object with NO EXTRA text, explanation, or markdown.
Every field must be present in your output. Use exactly the option strings as written.

When there is a 'choice' label after the name, Choose EXACTLY ONE of the options.

FIELDS AND VALID OPTIONS:

- valid (choice): ["YES", "NO"]

- Tri-lateral relationship (choice): ["1 - Yes", "2 - No", "3 - Not Applicable/Not Considered"]

- Managerial Role (choice): ["1 - Upper/Executive Management", "2 - Middle Management", "3 - Operational/Lower Management", "4 - No", "5 - Not Applicable/Not Considered"]

- committment (choice): ["1 - Written contract stating employee relationship", "2 - Oral contract stating employee relationship", "3 - Oral contract stating contractor relationship", "4 - Written contract stating contractor relationship", "5 - Written contract without relationship stated", "6 - Oral Contract without relationship stated", "7 - None", "8 - Not Applicable/Not Considered"]

- contract (choice): ["1 - Indefinite Full-Time Contract", "2 - Indefinite Part-Time Contract", "3 - Fixed Term Full-Time Contract", "4 - Fixed Term Part-Time Contract", "5 - Other", "6 - Not Applicable/Not Considered"]

- Termination Date (choice): ["1 - Yes", "2 - No", "3 - Not Applicable/Not Considered"]

- Minimum Pay or Work (choice): ["1 - Yes, work only", "2 - Yes, pay only", "3 - Yes, both pay and work", "4 - No", "5 - Not Applicable/Not Considered"]

- Insurance (choice): ["1 - Worker solely supplies liability insurance", "2 - Both Worker and Principal contributes to liability insurance", "3 - Hirer solely supplies liability insurance", "4 - Not Applicable/Not Considered"]

- Employment Benefits (choice): ["1 - Yes, all benefits normally given to employees", "2 - Yes, some benefits normally given to employees", "3 - No", "4 - Not Applicable/Not Considered"]

- Independence (choice): ["1 - Yes, and the Worker does in practice", "2 - Yes, the Worker is capable but DOES NOT in practice", "3 - No", "4 - Not Applicable/Not Considered"]

- Selection (choice): ["1 - Yes", "2 - No", "3 - Not Applicable/Not Considered"]

- Integration (choice): ["1 - Yes", "2 - No", "3 - Not Applicable/Not Considered"]

- Training (choice): ["1 - Yes, the Worker requires substantial training or education necessary for their job", "2 - Yes, the Worker requires some training or education ancillary for their job", "3 - No", "4 - Not Applicable/Not Considered"]

- Remuneration (choice): ["1 - Salary/Fixed Wages only without bonus", "2 - Salary/Fixed Wages with opportunity for incentives", "3 - Mix of salary and paid by commission/per task without incentives", "4 - Mix of salary and paid by commission/per task with opportunity for incentives", "5 - Paid by Commission/per task only without incentives", "6 - Paid by Commission/per task with opportunity for incentives", "7 - Not Applicable/Not Considered"]

- Payment Procedure (choice): ["1 - Yes", "2 - No", "3 - Not Applicable/Not Considered"]

- Opportunity for Loss (choice): ["1 - No risk", "2 - Minor risk (pays for some expenses/have some money invested, but not liable for losses to the Hirer's business)", "3 - High risk (pays for expenses/have money invested, AND liable for lossess to Hirer's business", "4 - Not Applicable/Not Considered"]

- Third Party Risk (choice): ["1 - Yes", "2 - No", "3 - Not Applicable/Not Considered"]

- Withholdings (choice): ["1 - Yes", "2 - No", "3 - Not Applicable/Not Considered"]

- Supervision (choice): ["1 - Yes", "2 - No", "3 - Not Applicable/Not Considered"]

- Disciplinary Action (choice): ["1 - Yes", "2 - No", "3 - Not Applicable/Not Considered"]

- Compliance (choice): ["1 - Yes, the hirer is required to verify that the worker is complying with local laws and regulation", "2 - No, the hirer is not required to ensure that the worker complies with local laws", "3 - No, the worker can refuse to disclosure these information", "4 - Not Applicable/Not Considered"]

- Allocation of Work (choice): ["1 - The Hirer assigns work, and it cannot be turned down by the Worker", "2 - The Hirer assigns work, but it can be turned down by the Worker", "3 - The Worker decides the work with approval from the Hirer", "4 - The Worker decides the work WITHOUT approval", "5 - Not Applicable/Not Considered"]

- Manner of Work (choice): ["1 - The Hirer gives instructions on how to perform the task and the Worker must comply", "2 - The Worker decides how to complete the task, but must comply with the requirements set by the Hirer or seek permission from the Hirer", "3 - The Worker is free to chose how to complete the task without restrictions", "4 - Not Applicable/Not Considered"]

- Work Hours (choice): ["1 - The Hirer sets the work schedule and work hours", "2 - The Hirer sets the work hours, but the Worker sets the schedule", "3 - The Worker sets the work hours, but the Hirer sets the schedule", "4 - The Worker sets their own schedule and work hours", "5 - Not Applicable/Not Considered"]

- Work Location (choice): ["1 - The Hirer sets the work location", "2 - The Worker sets the work location with permission from the Hirer", "3 - The Worker sets the work location without permission", "4 - Not Applicable/Not Considered"]

- Ownership of Resources (choice): ["1 - The Hirer owns/provides all of the resources", "2 - The Hirer owns/provides most of the resources, but not all the necessary ones", "3 - Both Parties own/provide equal proportions of resources", "4 - The Worker owns/provides most of the resources, but not all the necessary ones", "5 - The Worker owns/provides all of the resources", "6 - Not Applicable/Not Considered"]

- Distinction (choice): ["1 - High Level of Integration (Required to wear uniform, listed on company directories, held out as employee etc.)", "2 - Some Level of Integration (Business cards, email addresses, badges etc.)", "3 - No Integration (Nothing associating worker as part of business)", "4 - Not Applicable/Not Considered"]

- Exclusivity (choice): ["1 - No, the worker works exclusively for the Hirer", "2 - No, but the worker can others WITH permission from the Hirer", "3 - No, but the worker can work for others WITHOUT permission", "4 - Yes, the worker works equal amounts with the hirer and others", "5 - Yes, the worker does majority of work for others", "6 - Not Applicable/Not Considered"]

- Delegation (choice): ["1 - Yes, with permission from the Principal Hirer", "2 - Yes, without permission", "3 - Yes, Permission unstated", "4 - No", "5 - Not Applicable/Not Considered"]

- Union(choice): ["1 - Yes, with permission from the Principal Hirer", "2 - No", "3 - Not Applicable/Not Considered"]

- Worker Status (choice): ["1 - Employee", "2 - Indepedent Contractor"]

- LOE (String): Free text describing the length of employment (e.g. "14 years 8 months"). Write "Not Applicable/Not Considered" if not mentioned.

- Comments (String): Free text. Write any relevant observations about the case that do not fit the above fields. Write "" if none."""

        return f"{schema}\n\nHere are 10 annotated examples:\n\n{self.few_shot_examples}"

    TEXTAREA_FIELDS = {"LOE", "Comments"}
    CHOICES_FIELDS = {
        "valid", "Tri-lateral relationship", "Managerial Role", "committment", "contract",
        "Termination Date", "Minimum Pay or Work", "Insurance", "Employment Benefits",
        "Independence", "Selection", "Integration", "Training", "Remuneration",
        "Payment Procedure", "Opportunity for Loss", "Third Party Risk", "Withholdings",
        "Supervision", "Disciplinary Action", "Compliance", "Allocation of Work",
        "Manner of Work", "Work Hours", "Work Location", "Ownership of Resources",
        "Distinction", "Exclusivity", "Delegation", "Union", "Worker Status"
    }

    # HyperTextLabels span fields are intentionally omitted from model predictions.
    # Each Choices field has a companion *-text HyperTextLabels in annotation_config.xml,
    # and "relevant-text" covers Case Name, Case Number, Date, etc. Computing XPath +
    # character offsets in the HTML DOM reliably requires DOM traversal that is fragile
    # and out of scope for auto-annotation. Human annotators handle span highlighting
    # manually during review.
    def _parse_response(self, response: str) -> List[Dict]:
        regions = []
        data = json.loads(response)
        for field_name, value in data.items():
            if field_name in self.TEXTAREA_FIELDS:
                regions.append({
                    "type": "textarea",
                    "from_name": field_name,
                    "to_name": "text",
                    "value": {"text": [value]}
                })
            elif field_name in self.CHOICES_FIELDS:
                regions.append({
                    "type": "choices",
                    "from_name": field_name,
                    "to_name": "text",
                    "value": {"choices": [value]}
                })
        return regions

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        predictions = []
        params = {
            "provider": "openai",
            "api_key": self.OPENAI_KEY,
            "model": self.OPENAI_MODEL
        }
        for task in tasks:
            plain_text, source, year = self._get_task_data(task)
            messages = [
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": plain_text}
            ]
            response = gpt(messages, params)
            regions = self._parse_response(response[0])
            predictions.append(PredictionValue(result=regions, score=0.5))
        return ModelResponse(predictions=predictions)
