module {
  func.func @main(%arg0: tensor<10x78x76xf32>, %arg1: tensor<100x94x40xi16>, %arg2: tensor<1x94x1xi16>, %arg3: tensor<51x14x96x77x93x58xi1>) -> (tensor<51x14x96x77x93x58xi1>, tensor<20x702x228xf32>, tensor<51x14x96x77x93x58xi1>, tensor<20x702x228xi1>, tensor<100x94x80xi16>) {
    %t_0 = tosa.const_shape {values = dense<[ 1, 3, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %0 = tosa.tile %arg0, %t_0 : (tensor<10x78x76xf32>, !tosa.shape<3>) -> tensor<10x234x76xf32>
    %t_1 = tosa.const_shape {values = dense<[ 2, 3, 3 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %1 = tosa.tile %0, %t_1 : (tensor<10x234x76xf32>, !tosa.shape<3>) -> tensor<20x702x228xf32>
    %2 = tosa.bitwise_and %arg1, %arg2 : (tensor<100x94x40xi16>, tensor<1x94x1xi16>) -> tensor<100x94x40xi16>
    %3 = tosa.logical_not %arg3 : (tensor<51x14x96x77x93x58xi1>) -> tensor<51x14x96x77x93x58xi1>
    %4 = tosa.identity %1 : (tensor<20x702x228xf32>) -> tensor<20x702x228xf32>
    %5 = tosa.logical_left_shift %3, %3 : (tensor<51x14x96x77x93x58xi1>, tensor<51x14x96x77x93x58xi1>) -> tensor<51x14x96x77x93x58xi1>
    %t_6 = tosa.const_shape {values = dense<[ 1, 1, 2 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %6 = tosa.tile %2, %t_6 : (tensor<100x94x40xi16>, !tosa.shape<3>) -> tensor<100x94x80xi16>
    %7 = tosa.rsqrt %4 : (tensor<20x702x228xf32>) -> tensor<20x702x228xf32>
    %8 = tosa.floor %1 : (tensor<20x702x228xf32>) -> tensor<20x702x228xf32>
    %9 = tosa.bitwise_not %3 : (tensor<51x14x96x77x93x58xi1>) -> tensor<51x14x96x77x93x58xi1>
    %10 = tosa.greater %7, %7 : (tensor<20x702x228xf32>, tensor<20x702x228xf32>) -> tensor<20x702x228xi1>
    %11 = tosa.add %6, %6 : (tensor<100x94x80xi16>, tensor<100x94x80xi16>) -> tensor<100x94x80xi16>
    %12 = tosa.reverse %11 {axis = 0 : i32} : (tensor<100x94x80xi16>) -> tensor<100x94x80xi16>
    return %5, %8, %9, %10, %12 : tensor<51x14x96x77x93x58xi1>, tensor<20x702x228xf32>, tensor<51x14x96x77x93x58xi1>, tensor<20x702x228xi1>, tensor<100x94x80xi16>
  }
}
