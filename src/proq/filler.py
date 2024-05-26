from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys


import json
import os
from web_fillers import Filler
from .export import proq_to_json
from .utils import md2seek


class ProqFiller(Filler):
    def from_cookie(self, url, cookies):
        self.driver.get(url)
        for cookie in cookies:
            self.driver.add_cookie(cookie)
        self.driver.get(url)

    def load_data(self, proq_file):
        self.unit_name, self.proqs = proq_to_json(proq_file)
        for i, proq in enumerate(self.proqs):
            self.proqs[i]["statement"] = md2seek(proq["statement"])

    def open_dashboard():
        ...

    def check_value(self, name, state=True):
        try:
            checkbox = self.driver.find_element(By.NAME, name)
            checkbox_sibling = checkbox.find_element(
                By.XPATH, "./preceding-sibling::input[@type='checkbox']"
            )
            if (checkbox.get_dom_attribute("value") == "true") != state:
                checkbox_sibling.click()
        except Exception as e:
            print(f"Error checking value for {name}: {e}")

    def set_date_time(self, date_time="09/17/2023,14:30"):
        date, time = date_time.split(",")
        js_code = f"""
        document.querySelector('input[name="workflow:submission_due_date[0]"]').value = '{date}';
        document.querySelector('select[name="workflow:submission_due_date[1][0]"]').value = '{time.split(":")[0]}';
        document.querySelector('select[name="workflow:submission_due_date[1][1]"]').value = '{time.split(":")[1]}';
        """
        self.driver.execute_script(js_code)

    def set_testcase_content(self, testcase_no, content_type, is_public, content):
        if is_public:
            name_attribute = f"content:public_testcase[{testcase_no}]{content_type}"
        else:
            name_attribute = f"content:private_testcase[{testcase_no}]{content_type}"
        self.fill_text_area(name_attribute, content)

    def set_testcases(self, testcases):
        self.click_link("Add Public Test Case", len(testcases["public_testcases"]))
        self.click_link("Add Private Test Case", len(testcases["public_testcases"]))

        for i, t in enumerate(testcases["public_testcases"]):
            self.set_testcase_content(i, "input", True, t["input"])
            self.set_testcase_content(i, "output", True, t["output"])

        for i, t in enumerate(testcases["private_testcases"]):
            self.set_testcase_content(i, "input", False, t["input"])
            self.set_testcase_content(i, "output", False, t["output"])

    def set_code_content(self, template):
        textarea_mapping = {
            "prefix": "content:allowed_languages[0]prefixed_code",
            "template": "content:allowed_languages[0]code_template",
            "suffix": "content:allowed_languages[0]uneditable_code",
            "suffix_invisible": "content:allowed_languages[0]suffixed_invisible_code",
            "solution": "content:allowed_languages[0]sample_solution",
        }
        for k in template.keys():
            self.fill_text_area(textarea_mapping[k], template[k])

    def set_problem_statement(self, statement):
        html_button = self.driver.find_element(By.XPATH, "//button[text()='code']")
        html_button.click()
        self.fill_code_mirror(statement)

    def wait_till_outline_loaded(self):
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//*[@id="add_unit"]/button'))
        )

    def add_unit(self, unit_name):
        add_unit_btn = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//*[@id="add_unit"]/button'))
        )
        add_unit_btn.click()
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.LINK_TEXT, "Save"))
        )
        self.fill_input_by_name("title", unit_name)
        self.click_link("Save", wait=True)
        WebDriverWait(self.driver, 10).until(
            EC.text_to_be_present_in_element(
                (By.XPATH, '//*[@id="gcb-butterbar-message"]'), "Saved."
            )
        )
        self.driver.back()
        self.driver.refresh()

    def get_proq_urls(self, unit_name, proqs):
        xpath = f"//a[contains(text(), '{unit_name}')]/parent::*/parent::*/following-sibling::ol"
        unit = self.driver.find_element(By.XPATH, xpath)
        urls = [
            unit.find_element(
                By.XPATH, f".//a[contains(text(), '{proq}')]"
            ).get_attribute("href")
            for proq in proqs
        ]
        return urls

    def get_all_proq_urls(self, unit_name):
        xpath = f"//a[contains(text(), '{unit_name}')]/parent::*/parent::*/following-sibling::ol"
        unit = self.driver.find_element(By.XPATH, xpath)
        urls = [
            element.get_attribute("href")
            for element in unit.find_elements(
                By.XPATH, f".//div[contains(@class, 'name')]/a"
            )
        ]
        return urls

    def wait_till_proq_loaded(self, timeout):
        # wait till the save buttion is available
        WebDriverWait(self.driver, timeout).until(
            EC.presence_of_element_located((By.LINK_TEXT, "Save"))
        )

    def save_proq(self):
        self.click_link("Save")
        WebDriverWait(self.driver, 10).until(
            EC.text_to_be_present_in_element(
                (By.XPATH, '//*[@id="gcb-butterbar-message"]'), "Saved."
            )
        )

    def fill_data(self, data):
        while True:
            try:
                self.wait_till_proq_loaded(1)
                break
            except:
                try:
                    # for non secure sites.
                    self.click_by_xpath_text("button", "Continue to site", 1)
                except:
                    pass

        self.fill_input_by_name("title", data["title"])
        self.set_problem_statement(data["statement"])

        # defaults
        self.select_value("content:evaluator", "nsjail")
        self.select_value("workflow:evaluation_type", "Test cases")
        self.check_value("html_check_answers", True)
        self.check_value("content:ignore_presentation_errors", True)
        self.check_value("content:show_sample_solution", True)

        self.select_value("content:allowed_languages[0]language", data["lang"])
        self.set_date_time(data["deadline"])
        self.set_testcases(data["testcases"])
        self.set_code_content(data["code"])

        self.save_proq()

    def create_open_proqs(self):
        # wait till loaded
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located(
                (By.XPATH, '//ol[@class="course ui-sortable"]')
            )
        )
        last_unit = self.driver.find_element(
            By.XPATH, '//ol[@class="course ui-sortable"]/li[last()]'
        )
        button = last_unit.find_element(
            By.XPATH,
            './/form[@id="add_custom_unit_com.google.coursebuilder.programming_assignment"]/button',
        )
        for i in range(len(self.proqs)):
            ActionChains(self.driver).key_down(Keys.CONTROL).click(button).key_up(
                Keys.CONTROL
            ).perform()

    def create_proqs(self, create_unit=True):
        if create_unit:
            print(f"Creating unit: {self.unit_name}")
            self.add_unit(self.unit_name)
        else:
            print("Adding the proqs to the last unit")
        choice = input(f"{len(self.proqs)} proqs identified. Do you want to continue? (y/n)")
        if choice.lower() == "y":
            print("Creating the proqs.")
            self.create_open_proqs()
            input("Press any key after all windows opened to continue.")
            for i, proq in enumerate(self.proqs, 1):
                self.driver.switch_to.window(self.driver.window_handles[i])
                try:
                    print(f"setting {proq['title']}.")
                    self.fill_data(proq)
                except:
                    print(f"{proq['title']} is not set correctly.")
        else:
            print("Quiting.")
        self.driver.quit()