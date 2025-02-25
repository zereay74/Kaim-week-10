import React, { useState, useEffect } from "react";
import axios from "axios";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

function App() {
    const [data, setData] = useState([]);
    const [summary, setSummary] = useState({});
    const [forecast, setForecast] = useState([]);

    // Fetch data from Flask API
    useEffect(() => {
        axios.get("http://127.0.0.1:5000/api/data").then((res) => setData(res.data));
        axios.get("http://127.0.0.1:5000/api/summary").then((res) => setSummary(res.data));
        axios.get("http://127.0.0.1:5000/api/forecast").then((res) => setForecast(res.data));
    }, []);

    return (
        <div style={{ textAlign: "center", padding: "20px" }}>
            <h1>Brent Oil Price & Inflation Dashboard</h1>
            
            {/* Summary Stats */}
            <div style={{ display: "flex", justifyContent: "space-around", marginBottom: "20px" }}>
                <div><strong>Avg Price:</strong> ${summary.average_price}</div>
                <div><strong>Avg Inflation:</strong> {summary.average_inflation}%</div>
                <div><strong>Price Volatility:</strong> {summary.price_volatility}</div>
                <div><strong>Inflation Volatility:</strong> {summary.inflation_volatility}</div>
            </div>

            {/* Historical Trends */}
            <h2>Historical Trends</h2>
            <ResponsiveContainer width="90%" height={300}>
                <LineChart data={data}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="Year" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="Price" stroke="#8884d8" name="Brent Oil Price" />
                    <Line type="monotone" dataKey="Inflation_%" stroke="#82ca9d" name="Inflation Rate" />
                </LineChart>
            </ResponsiveContainer>

            {/* Forecast Trends */}
            <h2>Price & Inflation Forecast</h2>
            <ResponsiveContainer width="90%" height={300}>
                <LineChart data={forecast}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="Year" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="Predicted_Price" stroke="#FF5733" name="Predicted Price" />
                    <Line type="monotone" dataKey="Predicted_Inflation" stroke="#33FF57" name="Predicted Inflation" />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
}

export default App;
