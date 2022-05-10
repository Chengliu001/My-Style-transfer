import datetime

rows = await db.query("""
    with temp  as (
        select b.code,b.data_code,a.detail_router from md_data_list a, md_data_maindata_detail b where b.property_name =
         'A_AccountingCustomers' and a.data_id = '5' and a.code = b.data_code
    )
    select t.data_code,t.detail_router,md_data_list.code,md_data_list.detail_router as jx from md_data_list inner join 
    temp t on t.code = md_data_list.code and md_data_list.data_id = '6'and md_data_list.status = 0 and md_data_list.deleted = 0 
""")

current = datetime.datetime.now()
for row in rows:
    code = row['code']
    detail_router = row['detail_router']
    data_code = row['data_code']
    jx = row['jx']
    property_value = await db.query(
        f"select property_value from {detail_router} where code = '{data_code}' and property_name = 'C_DutyParagraph'"
    )
    thirdpart_code = await db.query(
        f"select thirdpart_code from {jx} where code = '{data_code}' and property_name = 'C_DutyParagraph'"
    )

    if len(property_value)> 0 and property_value[0]['property_value'] is not None:
        C_DutyParagraph = property_value[0]['property_value']

        if len(thirdpart_code)>0:
            await db.execute(
                f"insert into {jx} values ('{code}','{thirdpart_code[0]['thirdpart_code']}','C_DutyParagraph','{C_DutyParagraph}','{current}') on  duplicate key update property_value = '{C_DutyParagraph}',create_time='{current}'"
            )
        else:
            await db.execute(
                f"insert into {jx} values ('{code}','','C_DutyParagraph','{C_DutyParagraph}','{current}') on  duplicate key update property_value = '{C_DutyParagraph}',create_time='{current}'"
            )
        await db.execute(
            f"update md_data_list set version = version+1, modify_time='{current}' where code = '{code}'"
        )
        context = "纳税人识别号：{}；".format(C_DutyParagraph)
        await db.execute("""
                        insert into `md_action_log`
                        (object_id,object_code,operator,operate_type,operate_subject,context,create_time,remarks)
                        values ('6','{0}','2','更新','主数据','{1}','{2}','自动脚本5')
                    """.format(code, context, current))




