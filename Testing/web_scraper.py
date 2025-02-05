from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import os

chrome_driver_path = os.path.join(os.path.dirname(__file__), "chromedriver.exe")
service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=service)

driver.get("https://goseattleu.com/sports/mens-soccer")