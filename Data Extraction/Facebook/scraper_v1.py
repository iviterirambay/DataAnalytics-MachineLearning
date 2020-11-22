import argparse
import time
import json
import csv

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup as bs

from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys

import random

with open('facebook_credentials.txt') as file:
    EMAIL = file.readline().split('"')[1]
    PASSWORD = file.readline().split('"')[1]

def _login(browser, email, password):
    browser.get("http://facebook.com")
    browser.maximize_window()
    browser.find_element_by_name("email").send_keys(email)
    browser.find_element_by_name("pass").send_keys(password)
    browser.find_element_by_name('login').click()
    time.sleep(5)


def extract(page):
    option = Options()
    option.add_argument("--disable-infobars")
    option.add_argument("start-maximized")
    option.add_argument("--disable-extensions")

    # Pass the argument 1 to allow and 2 to block
    option.add_experimental_option("prefs", {
        "profile.default_content_setting_values.notifications": 1
    })

    # chromedriver should be in the same folder as file
    browser = webdriver.Chrome(executable_path="./chromedriver", options=option)
    _login(browser, EMAIL, PASSWORD)
    browser.get(page)
    actions = ActionChains(browser)

    AMOUNT_OF_TOTAL_SCROLLS = 1
    MIN_WAIT_TIME_BETWEEN_SCROLL = 1
    MAX_WAIT_TIME_BETWEEN_SCROLL = 4
    DUMMY_WAIT_TIME = 1

    # Hago scroll varias veces, de acuerdo a la variable AMOUNT_OF_TOTAL_SCROLLS.
    for _ in range(AMOUNT_OF_TOTAL_SCROLLS):
        actions.send_keys(Keys.SPACE).perform()
        print('Scroll...')
        time.sleep(random.randint(MIN_WAIT_TIME_BETWEEN_SCROLL, MAX_WAIT_TIME_BETWEEN_SCROLL))
    time.sleep(DUMMY_WAIT_TIME)

    posts = browser.find_elements_by_xpath("//*[contains(@class, 'du4w35lb k4urcfbm l9j0dhe7 sjgh65i0')]")

    print('Cantidad total de posts:', len(posts))
    c = 0
    for p in posts:
        c+=1
        print('============= Post', c, '=============')
        try:
            txt_box = p.find_element_by_xpath(".//*[contains(@class, 'kvgmc6g5 cxmmr5t8 oygrvhab hcukyx3x c1et5uql ii04i59q')]")
        except:
            continue
        txt = txt_box.find_elements_by_xpath("./div")
        for t in txt:
            print(t.text)
        try:
            more_btn = p.find_element_by_xpath(".//*[contains(@class, 'j83agx80 fv0vnmcu hpfvmrgz')]")
            more_btn.click()
            print('He dado click en un botón')
            time.sleep(2)
        except:
            print('No hubo botón')
            pass
        try:
            print('Comentarios')
            comments_box = p.find_element_by_xpath(".//*[contains(@class, 'cwj9ozl2 tvmbv18p')]")
            list_comments = comments_box.find_elements_by_xpath("./ul/li")
            print('Cantidad total de comentarios:', len(list_comments))
            for comment in list_comments:
                txt_box = comment.find_element_by_xpath(".//*[contains(@class, 'kvgmc6g5 cxmmr5t8 oygrvhab hcukyx3x c1et5uql')]")
                txt = txt_box.find_elements_by_xpath("./div")
                for t in txt:
                    print(t.text)
        except:
            continue
    print('Finalizaron los posts')
    time.sleep(180)
    exit()