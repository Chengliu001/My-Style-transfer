import cv2
import numpy as np


def image_processing(img1, img2):
    #  resize图片大小，入口参数为一个tuple，新的图片的大小
    img_1 = np.resize(img1, (520, 520))
    img_2 = np.resize(img2, (520, 520))
    #  处理图片后存储路径，以及存储格式
    return img_1, img_2


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    img1, img2: [0, 255]
    '''
    img_1 = img1
    img_2 = img2
    if not img1.shape == img2.shape:
        img1, img2 = image_processing(img_1, img_2)
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
def change_SSIM(score):
    socre_change = 1-(1-score)/2
    return socre_change


content = cv2.imread("C:/Users/admin/Desktop/used_img/content/forest2.jpg", 0)
style = cv2.imread("C:/Users/admin/Desktop/used_img/style/candy.jpg", 0)
sys_img = cv2.imread("C:/Users/admin/Desktop/used_img/sys/SANet/compare.jpg", 0)
ss1 = calculate_ssim(sys_img, content)
ss2 = calculate_ssim(sys_img, style)
print(ss1)
print(ss2)
print("内容相似度为:", change_SSIM(ss1))
print("风格相似度为:", change_SSIM(ss2))
import datetime

rows = await db.query("""
   with temp as (
       select 
           code, 
           max(case when property_name = 'C_Store_Status' then property_value else NULL end) as C_Store_Status,
                    max(case when property_name = 'C_Store_ActualOffDate' then property_value else NULL end) as C_Store_ActualOffDate ,
                    max(case when property_name = 'C_Store_ActualOnDate' then property_value else NULL end) as C_Store_ActualOnDate ,
                    max(case when property_name = 'C_Store_EstimateOffDate' then property_value else NULL end) as C_Store_EstimateOffDate ,
                    max(case when property_name = 'C_Store_EstimatedOnDate' then property_value else NULL end) as C_Store_EstimatedOnDate 

       from md_data_detail 
       group by code
       union all
       select 
           code, 
           max(case when property_name = 'C_Store_Status' then property_value else NULL end) as C_Store_Status,
                    max(case when property_name = 'C_Store_ActualOffDate' then property_value else NULL end) as C_Store_ActualOffDate ,
                    max(case when property_name = 'C_Store_ActualOnDate' then property_value else NULL end) as C_Store_ActualOnDate ,
                    max(case when property_name = 'C_Store_EstimateOffDate' then property_value else NULL end) as C_Store_EstimateOffDate ,
                    max(case when property_name = 'C_Store_EstimatedOnDate' then property_value else NULL end) as C_Store_EstimatedOnDate   
       from md_data_detail1
       group by code
       union all
       select 
           code, 
           max(case when property_name = 'C_Store_Status' then property_value else NULL end) as C_Store_Status,
                    max(case when property_name = 'C_Store_ActualOffDate' then property_value else NULL end) as C_Store_ActualOffDate ,
                    max(case when property_name = 'C_Store_ActualOnDate' then property_value else NULL end) as C_Store_ActualOnDate ,
                    max(case when property_name = 'C_Store_EstimateOffDate' then property_value else NULL end) as C_Store_EstimateOffDate ,
                    max(case when property_name = 'C_Store_EstimatedOnDate' then property_value else NULL end) as C_Store_EstimatedOnDate  
       from md_data_detail2
       group by code
       union all
       select 
           code, 
           max(case when property_name = 'C_Store_Status' then property_value else NULL end) as C_Store_Status ,
                    max(case when property_name = 'C_Store_ActualOffDate' then property_value else NULL end) as C_Store_ActualOffDate ,
                    max(case when property_name = 'C_Store_ActualOnDate' then property_value else NULL end) as C_Store_ActualOnDate ,
                    max(case when property_name = 'C_Store_EstimateOffDate' then property_value else NULL end) as C_Store_EstimateOffDate ,
                    max(case when property_name = 'C_Store_EstimatedOnDate' then property_value else NULL end) as C_Store_EstimatedOnDate 
       from md_data_detail3
       group by code
       union all
       select 
           code, 
           max(case when property_name = 'C_Store_Status' then property_value else NULL end) as C_Store_Status,
                    max(case when property_name = 'C_Store_ActualOffDate' then property_value else NULL end) as C_Store_ActualOffDate ,
                    max(case when property_name = 'C_Store_ActualOnDate' then property_value else NULL end) as C_Store_ActualOnDate ,
                    max(case when property_name = 'C_Store_EstimateOffDate' then property_value else NULL end) as C_Store_EstimateOffDate ,
                    max(case when property_name = 'C_Store_EstimatedOnDate' then property_value else NULL end) as C_Store_EstimatedOnDate  
       from md_data_detail4
       group by code
   )
   select 
       t.*,detail_router,thirdpart_code
   from md_data_list
   inner join temp t on
   t.code = md_data_list.code
   where md_data_list.data_id = 2 and md_data_list.status = 0 and md_data_list.deleted = 0 

""")

current = datetime.datetime.now()
for row in rows:
    status_old = row['C_Store_Status']
    status = None
    detail_router = row['detail_router']
    thirdpart_code = row['thirdpart_code']
    code = row['code']

    C_Store_ActualOffDate = row['C_Store_ActualOffDate']  # 实际闭店时间
    C_Store_ActualOnDate = row['C_Store_ActualOnDate']  # 实际开店时间
    C_Store_EstimateOffDate = row['C_Store_EstimateOffDate']  # 预计闭店时间
    C_Store_EstimatedOnDate = row['C_Store_EstimatedOnDate']  # 预计开店时间
    if C_Store_ActualOffDate:
        C_Store_ActualOffDate = datetime.datetime.strptime(row['C_Store_ActualOffDate'], '%Y-%m-%d')  # 实际闭店时间
    if C_Store_ActualOnDate:
        C_Store_ActualOnDate = datetime.datetime.strptime(row['C_Store_ActualOnDate'], '%Y-%m-%d')  # 实际开店时间
    if C_Store_EstimateOffDate:
        C_Store_EstimateOffDate = datetime.datetime.strptime(row['C_Store_EstimateOffDate'],
                                                             '%Y-%m-%d')  # 预计闭店时间
    if C_Store_EstimatedOnDate:
        C_Store_EstimatedOnDate = datetime.datetime.strptime(row['C_Store_EstimatedOnDate'],
                                                             '%Y-%m-%d')  # 预计开店时间
    # 开店1 停业2 闭店3 预闭店4 预开店5 装修6
    status_dict = {'1': "开店", '3': "闭店", '4': "预闭店", '5': "预开店"}
    if C_Store_ActualOffDate and current >= C_Store_ActualOffDate:
        status = '3'
    elif not C_Store_ActualOffDate and C_Store_EstimateOffDate and current >= C_Store_EstimateOffDate:
        status = '4'
    elif C_Store_ActualOffDate and C_Store_EstimateOffDate and C_Store_ActualOffDate > current >= C_Store_EstimateOffDate:
        status = '4'
    elif C_Store_ActualOffDate and not C_Store_EstimateOffDate and current < C_Store_ActualOffDate:
        status = '4'
    elif C_Store_ActualOnDate and current >= C_Store_ActualOnDate:
        status = '1'
    elif C_Store_EstimatedOnDate and not C_Store_ActualOnDate:
        status = '5'
    elif C_Store_EstimatedOnDate and C_Store_ActualOnDate and C_Store_ActualOnDate > current >= C_Store_EstimatedOnDate:
        status = '5'
    elif C_Store_EstimatedOnDate and C_Store_ActualOnDate and current <= C_Store_EstimatedOnDate:
        status = '5'
    elif C_Store_ActualOnDate and not C_Store_EstimatedOnDate and current < C_Store_ActualOnDate:
        status = '5'

    if status and status != status_old:
        await db.execute(f"""
                   insert into {detail_router} 
                   values ('{code}','{thirdpart_code}','C_Store_Status','{status}','{current}')
                   on duplicate key update property_value=values(property_value),create_time=values(create_time)
               """)
        await db.execute(
            f"update md_data_list set version = version+1, modify_time='{current}' where code = '{code}'"
        )
        context = "店铺状态：{0}；".format(status_dict[status])
        await db.execute(f"""
                   insert into `md_action_log`
                   (object_id,object_code,operator,operate_type,operate_subject,context,create_time,remarks)
                   values ('2','{code}','2','更新','主数据','{context}','{current}','自动脚本33（二批）')
               """)

