import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Typography from '@material-ui/core/Typography';
import Slider from '@material-ui/core/Slider';

const useStyles = makeStyles({
  root: {
    width: 300,
  },
});

const marks = [
    {
        value: '2/28/20',
        label: 'Feb 2020',
    },
    {
        value: '3/31/20',
        label: 'Mar 2020',
    },
    {
        value: '4/30/20',
        label: 'Apr 2020',
    },
    {
        value: '5/31/20',
        label: 'May 2020',
    },
    {
        value: '6/30/20',
        label: 'Jun 2020',
    },
    {
        value: '7/30/20',
        label: 'Jul 2020',
    },
];

const min = [
    {
        value: '1/22/20',
        label: 'Jan 2020',
    }
];

const max = [
    {
        value: '3/31/21',
        label: 'Mar 2021',
    }
];

const defVal = [
    {
        value: '6/30/20',
        label: 'Jun 2020',
    },
];

function valuetext(value) {
  return `${value}°C`;
}

export default function DiscreteSlider() {
  const classes = useStyles();

  return (
    <div className={classes.root}>
      <Typography id="discrete-slider" gutterBottom>
        Month Slider
      </Typography>
      <Slider
        defaultValue={defVal}
        getAriaValueText={valuetext}
        aria-labelledby="discrete-slider"
        valueLabelDisplay="auto"
        step={10}
        marks={marks}
        min={min}
        max={max}
      />
    </div>
  );
}