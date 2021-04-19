import React, { useState } from 'react';
import Header from './components/Header';
//import Slider from './components/Slider'
import Date from './components/Date';
import Map from './components/Map'


const App = () => {
  const [selectedDate, setSelectedDate] = useState('2021-02-04')

  return (
    <div className='container'>
      <Header />
      <Date setSelectedDate={setSelectedDate}/>
      <Map />
    </div>
  )
}
export default App