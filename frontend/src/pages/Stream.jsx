import Sidebar from "../components/Sidebar";
import PersonCard from "../components/PersonCard";
import CCTVStream from "../components/CCTVStream";
import { useEffect,useState } from "react";

export default function Stream(){

  const [persons,setPersons] = useState([]);
  const [selected,setSelected] = useState(null);

  useEffect(()=>{
    const interval = setInterval(async ()=>{
      const res = await fetch("http://localhost:8000/api/cctv/1/persons");
      const data = await res.json();
      setPersons(data.persons);
    },2000);

    return ()=>clearInterval(interval);
  },[]);

  return(
    <div style={{display:"flex"}}>
      <Sidebar/>

      <div style={{padding:"30px",flex:1}}>

        <CCTVStream/>

        {/* rest of UI */}
      </div>
    </div>
  )
}