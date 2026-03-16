import { useNavigate } from "react-router-dom";
import { useEffect, useRef } from "react";

export default function StreamCard() {
  const navigate = useNavigate();
  const imgRef = useRef(null);

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/cctv/1/ws");

    ws.binaryType = "arraybuffer";

    ws.onmessage = (event) => {
      const blob = new Blob([event.data], { type: "image/jpeg" });
      const url = URL.createObjectURL(blob);

      if (imgRef.current) {
        imgRef.current.src = url;
      }
    };

    ws.onclose = () => {
      console.log("WebSocket closed");
    };

    return () => {
      ws.close();
    };
  }, []);

  return (
    <div
      style={{
        width: "320px",
        background: "#fff",
        borderRadius: "12px",
        overflow: "hidden",
        boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
        cursor: "pointer"
      }}
      onClick={() => navigate("/stream/1")}
    >
      <img
        ref={imgRef}
        style={{
          width: "100%",
          height: "180px",
          objectFit: "cover",
          display: "block"
        }}
      />

      <div style={{ padding: "10px" }}>
        <h4>CCTV Ward 1</h4>
        <p style={{ color: "#999" }}>Live Monitoring</p>
      </div>
    </div>
  );
}