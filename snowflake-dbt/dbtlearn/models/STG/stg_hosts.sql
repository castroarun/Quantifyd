with raw_hosts as (
 SELECT * from {{ source('airbnb','hosts')}}
)

select id as host_ID,
	name as host_NAME,
	IS_SUPERHOST,
	CREATED_AT,
	UPDATED_AT
from raw_hosts