export default function PersonCard({person, onClick}) {
  return (
    <div
      onClick={()=>onClick(person)}
      style={{
        padding:"15px",
        background:"#fff",
        borderRadius:"10px",
        cursor:"pointer",
        boxShadow:"0 2px 8px rgba(0,0,0,0.1)"
      }}
    >
      <h4>{person.person_id}</h4>
      <p>Heart Rate: {person.heart_rate || "N/A"}</p>
    </div>
  );
}