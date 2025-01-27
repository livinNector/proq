from pathlib import Path

from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from .core import ProQ


def get_model(model_name):
    provider, model = model_name.split(":")
    match provider:
        case "open-ai":
            model = ChatOpenAI(model=model)
        case "groq":
            model = ChatGroq(model=model)

    return model


example_prompt = ChatPromptTemplate.from_messages(
    [("human", "{title}"), ("ai", "{proq}")]
)


def generate_proq(prompt, example_files, model="groq:gemma2-9b-it"):
    proqs = [
        ProQ.from_file(example_file, render_template=False)
        for example_file in example_files
    ]
    examples = [{"title": proq.title, "proq": proq.to_str()} for proq in proqs]
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Create a programming problem statement in a markdown format "
                "with the section Problem Statement, Solution, "
                "Public Test Cases and Private Test Cases, "
                "use an Yaml header for title and tags. "
                "The solution is annotated with template, sol, los and "
                "suffix_invisble tags. "
                "Use any jinja templates used in the examples. "
                "Retain the execution config from the first line of "
                "the solution code block."
                "Use the consistent same markdown format for the output, "
                "do not add any additional content."
                "Add relevant concept tags(atleast 3) for the new problem.",
            ),
            few_shot_prompt,
            ("human", "{prompt}"),
        ]
    )
    model = get_model(model)
    proq_template = (final_prompt | model).invoke({"prompt": prompt}).content

    un_rendered = ProQ.from_str(proq_template)
    corrected = ProQ.from_str(
        proq_template, Path(example_files[0]).parent, render_template=True
    ).correct_outputs()
    un_rendered.public_test_cases = corrected.public_test_cases
    un_rendered.private_test_cases = corrected.private_test_cases
    return un_rendered
