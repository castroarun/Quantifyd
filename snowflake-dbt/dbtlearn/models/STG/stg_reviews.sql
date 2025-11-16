with raw_reviews as (
SELECT * from {{ source('airbnb','reviews')}}
)

select LISTING_ID,
	DATE as listing_date,
	REVIEWER_NAME,
	COMMENTS,
	SENTIMENT
from raw_reviews