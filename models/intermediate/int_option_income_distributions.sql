select * from {{ ref('stg_rex__distributions') }}
union all
select * from {{ ref('stg_yieldmax__distributions') }}
