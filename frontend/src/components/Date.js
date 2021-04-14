const Date = ( {date} ) => {
    return (
        <div>
            <h1>{date}</h1>
        </div>
    )
}

Date.defaultProps = {
    date: '01/01/2021',
}

export default Date
