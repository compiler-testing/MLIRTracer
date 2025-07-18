module {
  func.func @main(%arg0: tensor<92x56x30xi32>, %arg1: tensor<86x69x92x77x14x98xf32>, %arg2: tensor<20x7x38x52x88xi1>, %arg3: tensor<1x1x38x52x1xi1>) -> (tensor<92x112x1xi32>, tensor<86x69x92x77x14x98xf32>, tensor<20x7x38x52x88xi1>) {
    %0 = tosa.abs %arg0 : (tensor<92x56x30xi32>) -> tensor<92x56x30xi32>
    %1 = tosa.ceil %arg1 : (tensor<86x69x92x77x14x98xf32>) -> tensor<86x69x92x77x14x98xf32>
    %2 = tosa.concat %0, %0 {axis = 1 : i32} : (tensor<92x56x30xi32>, tensor<92x56x30xi32>) -> tensor<92x112x30xi32>
    %3 = tosa.reduce_min %2 {axis = 2 : i32} : (tensor<92x112x30xi32>) -> tensor<92x112x1xi32>
    %4 = tosa.logical_xor %arg2, %arg3 : (tensor<20x7x38x52x88xi1>, tensor<1x1x38x52x1xi1>) -> tensor<20x7x38x52x88xi1>
    %5 = tosa.sigmoid %1 : (tensor<86x69x92x77x14x98xf32>) -> tensor<86x69x92x77x14x98xf32>
    %6 = tosa.pow %1, %5 : (tensor<86x69x92x77x14x98xf32>, tensor<86x69x92x77x14x98xf32>) -> tensor<86x69x92x77x14x98xf32>
    %7 = tosa.logical_not %4 : (tensor<20x7x38x52x88xi1>) -> tensor<20x7x38x52x88xi1>
    return %3, %6, %7 : tensor<92x112x1xi32>, tensor<86x69x92x77x14x98xf32>, tensor<20x7x38x52x88xi1>
  }
}
