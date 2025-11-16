{{
  config(
    materialized = 'incremental'
    )
}}
WITH stg_reviews AS (
select * from {{ ref('stg_reviews')}}
)
SELECT
  {{ dbt_utils.generate_surrogate_key(['listing_id', 'listing_date', 'reviewer_name', 'comments']) }} as review_id,
  *
FROM stg_reviews
WHERE comments is not null
{% if is_incremental() %}
  AND listing_date > (select max(listing_date) from {{ this }})
{% endif %}