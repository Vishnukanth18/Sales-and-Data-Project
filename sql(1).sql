create database RetailSalesData;
Use RetailSalesData;

create table Sales_Data_Transactions(
customer_id varchar(255),
trans_date varchar(255),
tran_amount int);


drop table Sales_Data_Transactions;

select * from Sales_Data_Transactions;

create table Sales_Data_Response(
customer_id varchar(255) primary key,
response int);
show tables;

load data infile 'C:\Users\DELL\OneDrive\Desktop\Internship Studio\archive/Retail_Data_Transactions.csv'
into table Sales_Data_Transactions
fields terminated by ','
lines terminated by '\n'
ignore 1 rows;


load data infile 'C:\Users\DELL\OneDrive\Desktop\Internship Studio\archive/Retail_Data_Response.csv'
into table Sales_Data_Response
fields terminated by ','
lines terminated by '\n'
ignore 1 rows;

explain select * from Sales_Data_Transactions where customer_id='CS5295';

create index idx_id on Sales_Data_Transactions(customer_id);

explain select * from Sales_Data_Transactions where customer_id='CS5295';