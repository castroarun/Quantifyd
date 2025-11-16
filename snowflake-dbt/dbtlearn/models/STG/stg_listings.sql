WITH raw_listings AS (
    SELECT * from {{ source('airbnb','listings')}}
)

select ID as LISTING_ID,
	NAME as LISTING_NAME,
    LISTING_URL,
	ROOM_TYPE,
	MINIMUM_NIGHTS,
	HOST_ID,
	PRICE as PRICE_STR,
	CREATED_AT,
	UPDATED_AT
FROM RAW_LISTINGS