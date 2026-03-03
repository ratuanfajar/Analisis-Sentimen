from google_play_scraper import reviews, Sort
import pandas as pd
from tqdm import tqdm
import time

def scrape_playstore_reviews(
    app_id='app.bpjs.mobile',
    target_count=4000,
    batch_size=500,
    filter_score=None
):
    all_reviews = []
    continuation_token = None

    print(f"Mulai scraping {target_count} ulasan...\n")

    for _ in tqdm(range(0, target_count, batch_size), desc="Mengambil review", unit="batch"):
        result, continuation_token = reviews(
            app_id,
            lang='id',
            country='id',
            sort=Sort.NEWEST,
            count=batch_size,
            filter_score_with=filter_score,
            continuation_token=continuation_token
        )

        all_reviews.extend(result)
        time.sleep(1)

        if continuation_token is None:
            break

    df = pd.DataFrame(all_reviews)
    print(f"\nSelesai! Total review: {len(df)}")

    return df