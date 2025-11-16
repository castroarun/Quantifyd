select * from {{ ref('fct_reviews')}} r, {{ref('dim_listings_cleansed')}} l
where l.listing_id = r.listing_id and r.listing_date < l.created_at
limit 10