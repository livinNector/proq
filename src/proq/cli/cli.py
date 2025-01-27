import os
from pathlib import Path
from typing import Literal

import fire
from termcolor import cprint

from proq.core import ProQ
from proq.evaluate_utils import ProqCheck
from proq.gen_ai_utils import generate_proq
from proq.utils import color_diff

from . import export


class ProqCli:
    """A Command-line suite for authoring Programming Questions.

    For help regarding individual commands use `proq [COMMAND] --help`.
    """

    def __init__(self) -> None:
        self.export = export.proq_export

    def create(
        self,
        output_file: str,
        lang: Literal["python", "java", "c"] = "python",
        n_public: int = 5,
        n_private: int = 5,
        force: bool = False,
    ):
        """Creates an empty proq file template with the given configuation.

        Args:
            output_file (str): Output file name
            lang (Literal["python","java","c"]) : Programming language used to
                create automatic execute configs.
                Possible values are python, java and c.
            n_public (int) : Number of public test cases
            n_private (int) : Number of private test cases
            force (bool) : Overwrite file if exists
        """
        if not force and os.path.isfile(output_file):
            raise FileExistsError(
                f"A file with the name '{output_file}' already exists."
            )

        ProQ.default_proq(
            lang=lang.lower().strip(), n_public=n_public, n_private=n_private
        ).to_file(output_file)

    def format(self, *proq_files: list[str]):
        """Formats the given files according to the proq template."""
        for proq_file in proq_files:
            ProQ.from_file(proq_file, render_template=False).to_file(proq_file)

    def correct(self, *proq_files: list[str]):
        """Corrects the test case outputs according to the solution."""
        for proq_file in proq_files:
            try:
                proq = ProQ.from_file(proq_file).correct_outputs(inplace=True)
                unrendered_proq = ProQ.from_file(proq_file, render_template=False)
                unrendered_proq.public_test_cases = proq.public_test_cases
                unrendered_proq.private_test_cases = proq.private_test_cases
                unrendered_proq.to_file(proq_file)

            except FileNotFoundError:
                print(f"{proq_file} is not a valid file.")

    def show_code(self, proq_file: str, render: bool = False):
        """Prints the whole solution where each part are highlighted.

        Args:
            proq_file (str): The proq file.
            render (bool): Whether to render the jinja template
        """
        proq = ProQ.from_file(proq_file, render_template=render)
        cprint(proq.solution.prefix, color="grey", end="")
        color_diff(proq.solution.template, proq.solution.solution)
        if proq.solution.suffix:
            cprint(proq.solution.suffix, color="grey", end="")
        if proq.solution.suffix_invisible:
            cprint(proq.solution.suffix_invisible, on_color="on_light_grey")

    def export_test_cases(self, proq_file, zip: bool = False):
        """Exports the test cases into a folder.

        Args:
            proq_file (str): The proq file
            zip (bool): Whether to zip archive instead of a folder.
        """
        proq = ProQ.from_file(proq_file)
        folder = Path(os.path.splitext(proq_file)[0])
        proq.export_test_cases(folder, zip)

    def evaluate(self, *files: str | os.PathLike, verbose=False, diff_mode=False):
        """Evaluates the testcases in the proq files locally.

        It uses the local installed compilers and interpreters
        to evalate the testcases.

        The config on how to execute the solution code is present
        in the first line of the code block in the solution.

        ```{lang_id} {filename} -r '{run_command}' -b '{build_command}'

        Args:
            files (str|PathLike): The file names of the proqs to be evaluated.
            verbose (bool): Whether to print the test results.
            diff_mode (bool):
                Whether to display expected-actual diff instead of separate
                expected and actual outputs
        """
        proq_checks: list[tuple[str, ProqCheck]] = []
        for file_path in files:
            if not os.path.isfile(file_path):
                print(f"{file_path} is not a valid file")
                continue
            print(f"Evaluating {file_path}")
            proq = ProQ.from_file(file_path)
            result = proq.evaluate(verbose=verbose, diff_mode=diff_mode)
            if verbose:
                print()
            proq_checks.append((file_path, result))

        n_proqs = len(proq_checks)
        cprint(
            f"Total of {n_proqs} proq{'s' if n_proqs>1 else ''} evaluated.",
            attrs=["bold"],
        )
        for file_path, proq_check in proq_checks:
            cprint(
                ("✓" if proq_check.solution_check else "✗") + " solution",
                "green" if proq_check.solution_check else "red",
                end=" ",
            )
            cprint(
                ("✓" if proq_check.template_check else "✗") + " template",
                "green" if proq_check.template_check else "red",
                end=" ",
            )
            print(os.path.relpath(file_path, os.curdir))

    def generate(self, prompt, *examples, output_file=None, model="groq:gemma2-9b-it"):
        proq = generate_proq(prompt, example_files=examples, model=model)
        if output_file is None:
            output_file = proq.title.lower().replace(" ", "_") + ".md"
        proq.to_file(output_file)


def main():
    fire.Fire(ProqCli(), name="proq")


if __name__ == "__main__":
    main()
