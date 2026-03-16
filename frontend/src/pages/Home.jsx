import Sidebar from "../components/Sidebar";
import StreamCard from "../components/StreamCard";

export default function Home(){

  return(
    <div style={{display:"flex"}}>
      <Sidebar/>

      <div style={{padding:"40px",flex:1}}>
        <h2>Channels</h2>

        <div style={{
          marginTop:"20px",
          display:"grid",
          gridTemplateColumns:"repeat(auto-fill,320px)",
          gap:"20px"
        }}>
          <StreamCard/>
        </div>
      </div>
    </div>
  )
}