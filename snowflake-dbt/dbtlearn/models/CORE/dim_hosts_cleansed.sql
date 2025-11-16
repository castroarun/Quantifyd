{{
config(
    materialized = 'incremental'
    )
}}
with stg_hosts as (
select * from {{ ref('stg_hosts')}}
)

select host_ID,
	NVL(host_NAME,'Anonymous') as host_NAME,
	IS_SUPERHOST,
	CREATED_AT,
	UPDATED_AT
from stg_hosts