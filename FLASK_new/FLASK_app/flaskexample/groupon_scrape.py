# https://github.com/nmaloof/WebScrapingProject/blob/master/groupon_reviews/group_rev_sel.py

from selenium import webdriver
import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas

def scraping(web_url):
    browser = webdriver.Chrome()
    browser.get(web_url)
    time.sleep(2)
    browser.find_element_by_xpath('//*[@id="nothx"]').click()
    time.sleep(2)
    browser.find_element_by_xpath('//*[@id="all-tips-link"]').click()
    time.sleep(2)
    df = pd.DataFrame(columns=['content'])
    i = 0
    output_file_name = './flaskexample/static/data/groupon_review.txt'
    save_to_file = open(output_file_name, 'w')
    
    
    while True:
        try:
            time.sleep(2)
            print("Scraping Page: " + str(i))
            reviews = browser.find_elements_by_xpath('//div[@class="tip-item classic-tip"]')
            next_bt = browser.find_element_by_link_text('Next')
            for review in reviews[3:]:
                content = review.find_element_by_xpath('.//div[@class="twelve columns tip-text ugc-ellipsisable-tip ellipsis"]').text
                save_to_file.write(content.strip('\n')+'\n')
            i += 1
            next_bt.click() 
        except:
            break

    return


