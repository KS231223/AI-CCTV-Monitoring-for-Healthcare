import { Link } from "react-router-dom";

export default function Sidebar() {
  return (
    <div style={{width:"220px",background:"#fff",height:"100vh",padding:"20px",borderRight:"1px solid #eee"}}>
      <h2>MedWatch</h2>

      <div style={{marginTop:"30px",display:"flex",flexDirection:"column",gap:"10px"}}>
        <Link className="btn btn-outline" to="/home">Channels</Link>
        <Link className="btn btn-outline" to="/registration">Registration</Link>
      </div>
    </div>
  );
}