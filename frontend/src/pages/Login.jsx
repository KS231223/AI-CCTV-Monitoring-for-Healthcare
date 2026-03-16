import { useNavigate } from "react-router-dom";
import { useState } from "react";

export default function Login(){

  const navigate = useNavigate();
  const [user,setUser] = useState("");
  const [pass,setPass] = useState("");

  function handleLogin(e){
    e.preventDefault();
    navigate("/home");
  }

  return(
    <div style={{display:"flex",justifyContent:"center",alignItems:"center",height:"100vh"}}>
      <form onSubmit={handleLogin} style={{width:"300px",display:"flex",flexDirection:"column",gap:"10px"}}>
        <h2>Login</h2>

        <input placeholder="username" onChange={e=>setUser(e.target.value)}/>
        <input placeholder="password" type="password" onChange={e=>setPass(e.target.value)}/>

        <button className="btn btn-primary">Login</button>
      </form>
    </div>
  )
}