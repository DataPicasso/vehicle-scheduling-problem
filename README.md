

---

## **📌 Route Optimization Project using Clustering and TSP Algorithms**  
**Author:** Pedro Miguel Figueroa Domínguez  

### **📖 Introduction**  
This project is designed to **optimize visit routes** to different geographical locations using **clustering algorithms and the Traveling Salesman Problem (TSP) solution**, if you want to try it click [HERE](https://bestroutes.streamlit.app/). The process consists of two main steps:  

1. **Clustering Points:**  
   - A clustering algorithm groups locations into clusters with a **maximum of 281 points** each.  
   - These clusters are assigned to different **agents** to distribute the workload.  

2. **Route Optimization (TSP):**  
   - The **Nearest Neighbor Algorithm** is used to determine the optimal visiting order within each cluster.  
   - The route is sorted based on the **shortest distance between points**.  

### **⚙️ Prerequisites**  
Before running the notebook, install the required Python libraries:  

```python
!pip install pandas googlemaps ortools folium openpyxl
pip install folium pandas numpy scikit-learn geopy
```
 

### **🌍 Getting Coordinates (If Not Available in Your Data)**  
If your dataset does not contain **latitude and longitude coordinates**, you can extract them using the **Google Maps API**.  

#### **Steps to Obtain Coordinates**  

1. **Download an Address Dataset (if needed)**  
   - If you only have business names and addresses, your file should have structured address information.  
   - Recommended format: CSV or Excel with columns like **Business Name, Address, City, Country**.  

2. **Get a Google Maps API Key**  
   - Go to [Google Cloud Console](https://console.cloud.google.com/)  
   - Enable the **Geocoding API**  
   - Generate an **API Key**  

3. **Insert the API Key into the Code**  
   - The script will use this API key to query Google Maps and retrieve latitude/longitude.  

4. **Run the Script to Convert Addresses to Coordinates**  
   - The code will process each address and store the coordinates for later use.  

### **🚀 How to Use This Project**  

1. **Load the Data**  
   - Ensure your dataset includes either **coordinates** or **addresses** (if you plan to extract them).  
   - If missing coordinates, use the Google Maps API method to obtain them.  
   - The script will automatically load and process the dataset into a **DataFrame**.  

2. **Run the Route Optimization Algorithm**  
   - The script will create clusters of **maximum 281 points** each.  
   - It will then apply the **TSP algorithm** to determine the optimal visiting order.  

3. **Generate the CSV File and Visualize the Route**  
   - Enter the **agent number** to generate a **route table**.  
   - The script will generate a CSV file with the optimized route.  
   - A **map will be displayed**, showing the numbered points in the correct order.  

### **📌 Key Features**  
✔ **Efficient clustering** to distribute locations into manageable groups.  
✔ **Route optimization** using the **Nearest Neighbor TSP algorithm**.  
✔ **Automatic CSV generation** with sorted visiting orders.  
✔ **Interactive map visualization** with numbered points.  
✔ **Automatic coordinate retrieval** if not available.  

### **🎯 Expected Results**  
- **A well-organized table** displaying the optimal visiting order.  
- **An interactive map** showing points numbered in sequence.  
- **A CSV file** with the ordered route for each agent.  

### **📊 Excel File Structure**  
The Excel file used in this project must have the following structure:

- The **sheet** must be called `Sheet1`.  
- The **columns** should start from the **2nd row** with these headers:

|   | A              | B       | C   | D       | E          | F         |
|---|----------------|---------|-----|---------|------------|-----------|
| 1 |                |         |     |         |            |           |
| 2 | **Nombre Comercial** | **Calle** | **No.** | **Sector** | **Municipio** | **Provincia** |
| 3 | Example Name    | Example St. | 123 | Sector 1 | City 1     | Province 1 |
| 4 | Example Name 2  | Another St. | 456 | Sector 2 | City 2     | Province 2 |
| 5 | ...             | ...     | ... | ...     | ...        | ...       |

- Data should start from **row 3**. The **columns** are as follows:
  - **Nombre Comercial** (Business Name)  
  - **Calle** (Street)  
  - **No.** (Street Number)  
  - **Sector** (Sector)  
  - **Municipio** (City)  
  - **Provincia** (Province)  

---

💡 **This project streamlines visit planning and route optimization, ensuring efficiency in travel.** 🚀
