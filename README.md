# Instructions to install

Install packages from requirements.txt: <br>
pip install -r requirements.txt <br>

Download the SSL certificate from TiDB: <br>
Connect > Connect with SQLAlchemy > Download CA Cert <br>
Rename it to cert.pem <br>
Paste it in the same folder as the program <br>

Create a .env file like so: <br>
TIDB_USER=<your_username> <br>
TIDB_PASSWORD=<your_password> <br>
TIDB_HOST=<your_host> <br>
TIDB_PORT=4000 <br>
TIDB_DATABASE=test <br>
OPENAI_API_KEY=<your_openai_api_key> <br>

Run the program: <br>
streamlit run main.py <br>