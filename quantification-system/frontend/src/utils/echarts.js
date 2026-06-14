import { init, use } from 'echarts/core'
import { CandlestickChart, BarChart, ScatterChart, LineChart } from 'echarts/charts'
import {
  GridComponent, DataZoomComponent, TooltipComponent,
  MarkPointComponent, MarkLineComponent, LegendComponent,
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'

use([
  CandlestickChart, BarChart, ScatterChart, LineChart,
  GridComponent, DataZoomComponent, TooltipComponent,
  MarkPointComponent, MarkLineComponent, LegendComponent,
  CanvasRenderer,
])

export default { init }
