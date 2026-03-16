import { useEffect, useRef } from "react";

export default function CCTVStream() {
  const imgRef = useRef(null);

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/api/cctv/1/ws");

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
    <img
      ref={imgRef}
      style={{ width: "800px", borderRadius: "10px" }}
    />
  );
}