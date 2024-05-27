from ..parse import proq_to_json
from .utils import to_seek
from playwright.async_api import expect
from .filler import Filler
import asyncio


class SeekFiller(Filler):
    def load_data(self, proq_file):
        self.unit_name, self.proqs = proq_to_json(proq_file)
        for proq in self.proqs:
            proq["statement"] = to_seek(proq)

    async def goto_course_dashboard(self,course_code,domain="onlinedegree"):
        await self.page.goto(
            f"https://backend.seek.{domain}.iitm.ac.in/modules/firebase_auth/login?continue=https://backend.seek.{domain}.iitm.ac.in/{course_code}/dashboard"
        )
        await self.page.wait_for_url(f"https://backend.seek.{domain}.iitm.ac.in/{course_code}/dashboard")

    async def continue_blocked_page(self,page):
        try:
            await page.wait_for_load_state("domcontentloaded",timeout=20000)
            await page.get_by_text("Continue to site").dispatch_event("click",timeout=3000)
        except Exception as e:
            print(e)

    async def add_unit(self, unit_name):
        await self.page.locator('#add_unit > button').click()
        await self.continue_blocked_page(self.page)
        await self.page.locator("[name='title']").fill(unit_name)
        await self.page.get_by_role("link",name="Save").click()
        await expect(self.page.get_by_text("Saved.")).to_be_visible()
        await self.page.go_back()
        await self.page.reload()

    async def create_proqs_in_unit(self, unit_name=None,proqs=None):
        """Creates the proqs in the given unit, if unit not given 
        uses the last unit. If the proqs with the same name exists 
        in the unit edits those instead of creating new proqs"""

        await expect(self.page.locator('css=ol.course.ui-sortable')).to_be_visible(timeout=20_000)
        last_unit = self.page.locator('css=ol.course.ui-sortable > li:last-child')
        new_proq_button = last_unit.locator(
            'form#add_custom_unit_com\\.google\\.coursebuilder\\.programming_assignment > button'
        )
        async def click_and_wait(element):
            async with self.context.expect_page():
                await element.click(modifiers=["ControlOrMeta"])
        asyncio.gather(*(click_and_wait() for i in range(len(self.proqs))))
            

    async def set_problem_statement(self, page, statement):
        # click html code button
        await page.locator("button:text('code')").click()
        text_area = page.locator(".CodeMirror textarea")
        # delete existing problem statement
        await text_area.press("Control+a")
        await text_area.press("Delete")
        # enter problem statement
        await text_area.fill(statement)

    async def check_value(self,page, name,value =True):
        check_box = page.locator(f'[name="{name}"]').locator("xpath=./preceding-sibling::input[@type='checkbox']")
        if value == True:
            await check_box.check()
        else:
            await check_box.uncheck()

    async def set_date_time(self, page, date_time="09/17/2023,14:30"):
        date, time = date_time.split(",")
        js_code = f"""
        document.querySelector('input[name="workflow:submission_due_date[0]"]').value = '{date}';
        document.querySelector('select[name="workflow:submission_due_date[1][0]"]').value = '{time.split(":")[0]}';
        document.querySelector('select[name="workflow:submission_due_date[1][1]"]').value = '{time.split(":")[1]}';
        """

        await page.evaluate(js_code)

    async def delete_existing_testcases(self, page, public=True):
        testcase_items = await page.locator(f"#{'public' if public else 'private'}_tests_reg").get_by_text("Delete").all()
        for testcase_item in testcase_items[::-1]:
            await testcase_item.click()

    async def set_testcase_content(self, page, testcase_no, content_type, is_public, content):
        name_attribute = f"content:{'public' if is_public else 'private'}_testcase[{testcase_no}]{content_type}"
        await page.locator(f"textarea[name='{name_attribute}']").fill(content)


    async def set_testcases(self, page, testcases):

        await self.delete_existing_testcases(page, public=True)
        await self.delete_existing_testcases(page, public=False)

        for i in range(len(testcases["public_testcases"])):
            await page.get_by_text("Add Public Test Case").click()
        
        for i in range(len(testcases["private_testcases"])):
            await page.get_by_text("Add Private Test Case").click()

        for i, t in enumerate(testcases["private_testcases"]):            
            await self.set_testcase_content(page, i, "input", True, t["input"]),
            await self.set_testcase_content(page, i, "output", True, t["output"])

        for i, t in enumerate(testcases["private_testcases"]):
            await self.set_testcase_content(page, i, "input", False, t["input"]),
            await self.set_testcase_content(page, i, "output", False, t["output"])
            

    async def set_code_content(self, page, template):
        textarea_mapping = {
            "prefix": "content:allowed_languages[0]prefixed_code",
            "template": "content:allowed_languages[0]code_template",
            "suffix": "content:allowed_languages[0]uneditable_code",
            "suffix_invisible": "content:allowed_languages[0]suffixed_invisible_code",
            "solution": "content:allowed_languages[0]sample_solution",
        }
        for k in template.keys():
            await page.locator(f"textarea[name='{textarea_mapping[k]}']").fill(template[k])

    async def save_proq(self, page)->bool:
        """
        Saves the proq and rerturns the status.
        """
        await page.get_by_role("link",name="Save").click()
        saved_message = page.get_by_text("Saved.")
        failed_message = page.get_by_text("Please validate testcases")
        await expect(saved_message.or_(failed_message)).to_be_visible(timeout=20000)
        return not (await failed_message.is_visible())
            

    async def fill_data(self, data, page):
        # Seek configs
        seek_config = data["seek_config"]
        await page.locator('[name="content:evaluator"]').select_option(seek_config["evaluator"])
        await page.locator('[name="workflow:evaluation_type"]').select_option(seek_config["evaluator_type"])

        await self.set_date_time(page, seek_config["deadline"])

        await self.check_value(page,"html_check_answers", seek_config["allow_compile"])
        await self.check_value(page,"content:ignore_presentation_errors", seek_config["ignore_presentation_error"])
        await self.check_value(page,"content:show_sample_solution", seek_config["show_sample_solution"])
        
        await page.locator(f'[name="is_draft"]').select_option(label = "Public" if seek_config["is_public"] else "Private")

        await page.locator(f'[name="content:allowed_languages[0]language"]').select_option(seek_config["lang"])
        await page.locator(f'[name="content:allowed_languages[0]filename"]').fill(seek_config.get("solution_file",""))

        # Proq details
        await page.locator(f'[name="title"]').fill(data["title"])
        await self.set_problem_statement(page, data["statement"])

        await self.set_testcases(page, data["testcases"])
        await self.set_code_content(page, data["code"])

        return await self.save_proq(page)

    async def get_proq_urls(self, unit_name, proqs):
        xpath = f"//a[contains(text(), '{unit_name}')]/parent::*/parent::*/following-sibling::ol"
        unit = self.page.query_selector(xpath)
        urls = [
            unit.query_selector(f'.//a[contains(text(), "{proq}")]').get_attribute("href")
            for proq in proqs
        ]
        return urls

    def get_all_proq_urls(self, unit_name):
        xpath = f"//a[contains(text(), '{unit_name}')]/parent::*/parent::*/following-sibling::ol"
        unit = self.page.query_selector(xpath)
        urls = [
            element.get_attribute("href")
            for element in unit.query_selector_all('.name a')
        ]
        return urls

    async def create_proqs_in_unit(self, unit_name):
        # TODO: implement create unit if not exists
        unit = self.page.locator("ol.course.ui-sortable>li",has_text=unit_name).locator("ol.unit.ui-sortable")
        proq_links = await unit.get_by_role("link").filter(has_not_text="open_in_new").all()
        proq_links = {
            (await proq_link.inner_text()).strip() : proq_link
            for proq_link in proq_links
        }

        new_proq_button = unit.locator(
            'form#add_custom_unit_com\\.google\\.coursebuilder\\.programming_assignment > button'
        )

        for proq in self.proqs:
            element = proq_links.get(proq["title"].strip(), new_proq_button)
            await element.click(modifiers=["ControlOrMeta"])

        while len(self.context.pages) != len(self.proqs)+1:
            await asyncio.sleep(1)

        for page in self.context.pages[1:]:
            await self.continue_blocked_page(page)


        statuses = await asyncio.gather(*(
            self.fill_data(proq,page) 
            for proq, page in zip(self.proqs, self.context.pages[1:])
        ))
        return {proq["title"]: status for proq, status in zip(self.proqs, statuses)}


    async def create_proqs(self, create_unit=True):
        print(f"Unit Name: {self.unit_name}")
        print("Below proqs are identified.")
        print(*(f"{i}. {proq['title']}" for i, proq in enumerate(self.proqs,1)),sep="\n")
        choice = input("Do you want to continue? (y/n)")
        if choice.lower() == "y":
            print(f"Creating the proqs in {self.unit_name}")
            statuses = await self.create_proqs_in_unit(self.unit_name)
            for title ,status in statuses.items():
                print(title,"Success" if status else "Failed")
        else:
            print("Quiting.")
        await self.context.close()
        await self.playwright.stop()