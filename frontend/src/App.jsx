import {BrowserRouter,Routes,Route} from "react-router-dom"

import Login from "./pages/Login"
import Home from "./pages/Home"
import Stream from "./pages/Stream"
import Registration from "./pages/Registration"

export default function App(){

  return(
    <BrowserRouter>

      <Routes>
        <Route path="/" element={<Login/>}/>
        <Route path="/home" element={<Home/>}/>
        <Route path="/stream/:id" element={<Stream/>}/>
        <Route path="/registration" element={<Registration/>}/>
      </Routes>

    </BrowserRouter>
  )
}