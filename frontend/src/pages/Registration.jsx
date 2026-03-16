import Sidebar from "../components/Sidebar";
import { useState } from "react";

export default function Registration() {
  const [fileName, setFileName] = useState("No file selected");

  const handleFileChange = (e) => {
    if (e.target.files.length > 0) {
      setFileName(e.target.files[0].name);
    }
  };

  return (
    <div style={{ display: "flex" }}>
      <Sidebar />

      <div style={{ padding: "40px", flex: 1 }}>
        <h2>Patient Registration</h2>
        <p>This page will be used to register patients.</p>

        <form
          style={{
            marginTop: "20px",
            display: "flex",
            flexDirection: "column",
            gap: "20px",
            width: "360px",
          }}
        >
          <input
            placeholder="Patient Name"
            className="form-input"
          />
          <input placeholder="Height" className="form-input" />
          <input placeholder="Age" className="form-input" />

          {/* File Upload */}
          <label className="file-upload">
            <input type="file" onChange={handleFileChange} />
            <div className="file-box">
              <p>{fileName}</p>
              <span>📁 Upload Picture</span>
            </div>
          </label>

          <button className="btn btn-primary btn-full">Register</button>
        </form>
      </div>
    </div>
  );
}