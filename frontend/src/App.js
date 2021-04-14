import Header from './components/Header'
import Date from './components/Date'
import Map from './components/Map'

const App = () => {
  return (
    <div className='container'>
      <Header />
      <Date date='02/27/2021'/>
      <Map />
    </div>
  )
}
export default App