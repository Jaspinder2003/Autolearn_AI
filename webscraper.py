from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import requests
import os

"""
Serches the links of the image from google

Parameters:
searchPrompt (str): The prompt to be serched on google
numberOfImages (int): The number of images to be searched

Returns: 
image_urls(str set): The set of the found Urls of images
"""
def imageSearch(searchPrompt, numberOfImages):
    finalPrompt = searchPrompt.replace(" ", "+")
    
    # Set up Selenium WebDriver
    driver = webdriver.Chrome()
    driver.get(f"https://www.google.com/search?q={finalPrompt}&tbm=isch")
    time.sleep(2)  # Allow page to load

    # Scroll down to load more images
    for _ in range(3):  
        driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)

    # Find all clickable image elements
    image_elements = driver.find_elements(By.CLASS_NAME, "mNsIhb")
    
    image_urls = set()  # Use a set to store unique image URLs
    images_downloaded = 0  # Keep track of how many images we have downloaded

    while images_downloaded < numberOfImages and len(image_elements) > 0:  
        for i, img in enumerate(image_elements):  
            try:
                img.click()  # Click to open image
                
                # Wait until the image is visible
                full_img = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "img.sFlh5c"))
                )
                
                # Get the URL of the image
                img_url = full_img.get_attribute("src")
                if img_url and img_url.startswith("http"):
                    if img_url not in image_urls:  # Ensure it's unique
                        image_urls.add(img_url)
                        images_downloaded += 1
                    else:
                        print(f"Duplicate image URL found, skipping image {i+1}")
                
            except Exception as e:
                print(f"Error with image {i+1}: {e}")
            
            if images_downloaded >= numberOfImages:  # Exit if we have downloaded enough images
                break

        # If not enough images were found, scroll down to load more images and retry
        if images_downloaded < numberOfImages:
            driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
            time.sleep(0.5)
            image_elements = driver.find_elements(By.CLASS_NAME, "mNsIhb")

    driver.quit()

    return image_urls

    #To make this whole thing one function Uncomment the last comment
    # Download images
    #save_images(image_urls, searchPrompt)

"""
Saves images from the set of links from the web

Parameters:
searchPrompt (str): The prompt to be serched on google
image_urls (str set): The set of img links which need to be downloaded

Returns: 
Saves images in the scraped_images folder
"""
def save_images(image_urls, searchPrompt):
    main_folder_name = "scraped_images"
    folder_name = searchPrompt.replace(" ", "_")
    os.makedirs(f"{main_folder_name}/{folder_name}", exist_ok=True)
    it=0
    for i, img_url in enumerate(image_urls):
        try:
            response = requests.get(img_url)
            response.raise_for_status()  # Raise an exception for invalid URLs
            with open(f"{main_folder_name}/{folder_name}/{searchPrompt}_{i+1}.jpg", "wb") as file:
                file.write(response.content)
            it+=1
        except requests.exceptions.RequestException as e:
            print(f"Error downloading image {i+1}: {e}")
    print(f"Downloaded {it} images of {searchPrompt}")




"""# Test 1
prompt="Funny Animal"
url=imageSearch(prompt, numberOfImages=10)
save_images(url,prompt)


#Test 2
prompt="Cars"
url=imageSearch(prompt, numberOfImages=10)
save_images(url,prompt)

#Test 3
prompt="Trains"
url=imageSearch(prompt, numberOfImages=10)
save_images(url,prompt)"""


