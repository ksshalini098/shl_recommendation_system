import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

BASE_URL = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

def get_all_assessment_links():
    links = set()
    start = 0

    while True:
        url = f"{CATALOG_URL}?start={start}"
        print(f"Scraping catalog page starting at {start}...")

        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.text, "html.parser")

        cards = soup.select("a[href*='/products/product-catalog/view/']")

        if not cards:
            break

        for card in cards:
            href = card.get("href")

            if not href:
                continue

            # Filter only Individual Test Solutions
            unwanted_keywords = [
                "job-profiling",
                "framework",
                "guide",
                "interview",
                "pre-packaged",
                "solution"
            ]

            if not any(word in href.lower() for word in unwanted_keywords):
                full_url = BASE_URL + href
                links.add(full_url)

        start += 12
        time.sleep(1)

    return list(links)


def extract_test_type(soup):
    """
    Extract Test Type (K, P, A etc.)
    """
    text = soup.get_text(separator=" ").lower()

    if "test type" in text:
        # Try structured extraction
        labels = soup.find_all("strong")
        for label in labels:
            if "test type" in label.text.lower():
                parent = label.parent.get_text(strip=True)
                return parent.replace("Test Type:", "").strip()

    return ""


def scrape_assessment(url):
    try:
        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.text, "html.parser")

        title_tag = soup.find("h1")
        name = title_tag.text.strip() if title_tag else ""

        meta = soup.find("meta", {"name": "description"})
        description = meta["content"].strip() if meta else ""

        test_type = extract_test_type(soup)

        # Ignore Pre-packaged category if present
        page_text = soup.get_text().lower()
        if "pre-packaged job solution" in page_text:
            return None

        return {
            "name": name,
            "url": url,
            "description": description,
            "test_type": test_type
        }

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None


def main():
    print("Getting all assessment links...")
    links = get_all_assessment_links()

    print(f"\nTotal links found: {len(links)}")

    data = []

    for i, link in enumerate(links):
        print(f"Scraping {i+1}/{len(links)}")
        result = scrape_assessment(link)

        if result and result["name"]:
            data.append(result)

        time.sleep(0.5)

    df = pd.DataFrame(data)

    df.drop_duplicates(subset=["url"], inplace=True)
    df = df[df["name"] != ""]
    df = df.reset_index(drop=True)

    df.to_csv("data/assessments.csv", index=False)

    print("\nDONE")
    print(f"Total assessments saved: {len(df)}")
    print("Saved to data/assessments.csv")


if __name__ == "__main__":
    main()