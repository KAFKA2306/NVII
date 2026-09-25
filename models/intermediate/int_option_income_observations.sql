select * from {{ ref('stg_rex__observations') }}
union all
select * from {{ ref('stg_yieldmax__observations') }}
