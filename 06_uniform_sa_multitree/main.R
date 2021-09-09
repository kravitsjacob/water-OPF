

require(mvpart)
require(TeachingDemos)
require(RColorBrewer)


pathto_data = 'G:\\My Drive\\Documents (Stored)\\data_sets\\Water_OPF_GS_V3_io'
pathto_results = file.path(pathto_data, 'output', 'results.csv')
param_labs = c('Withdraw Weight ($/Gallon)', 'Consumption Weight ($/Gallon)', 'Uniform Water Factor', 'Uniform Loading Factor')
param_display_labs = c('Withdraw.Weight', 'Consumpion.Weight', 'Uniform.Water.Factor', 'Uniform.Load.Factor')
obj_labs = c('Total Cost ($)', 'Generator Cost ($)', 'Water Withdraw (Gallon)', 'Water Consumption (Gallon)')
scale_obj_labs = c('Total.Cost.Scaled', 'Generator.Cost.Scaled', 'Water.Withdraw.Scaled', 'Water.Consumption.Scaled')
scale_obj_display_labs = c('Total Cost (Scaled)', 'Generator Cost (Scaled)', 'Water Withdraw (Scaled)', 'Water Consumption (Scaled)')
colors = brewer.pal(length(scale_obj_display_labs), 'Set2')

init = function(){
  # Import
  df = read.table(pathto_results, header=TRUE, sep=',', check.names=FALSE)
  # Scale objectives to between 0 and 1
  df[, scale_obj_labs] = apply(df[,obj_labs], 2, function(x) (x-min(x))/(max(x) - min(x)))
  # Rename Parameters
  df[, param_display_labs] = df[, param_labs]
  return(df)
}


getModelHyperparameters = function(df){
  # Initialized Vars
  stop_rule = 0.001
  #form = (data.matrix(df[,scale_obj_labs])) ~ (get(dec_labs[1])+get(dec_labs[2])+get(coef_labs[1])+ get(coef_labs[2]))
  form = makeform(df, ycol=match(scale_obj_labs, colnames(df)), xcol=match(param_display_labs, colnames(df)))
  # Cross-Validated Errors
  mrt_error = mvpart(form, data = df, pretty = T, xv = "min", minauto = F,
                     which = 4, bord = T, uniform = F, text.add = T, branch = 1,
                     xadj = .7, yadj = 1.2, use.n = T, margin = 0.2, 
                     keep.y = F, bars = F, all.leaves = F,
                     control = rpart.control(cp = stop_rule, xval = 10), #10-fold cross validation
                     plot.add = F)
  # Plotting Cross-Validated Errors
  plotcp(mrt_error, upper='size', minline=FALSE, resub.err = FALSE)
  plot_performance = recordPlot()
  # Observation from Plot
  size = 7
  # Get Corresponding Complexity Parameter
  full.table = as.data.frame(mrt_error$cptable)
  cp = full.table$CP[full.table$nsplit == (size-1)]
  # Prepare for Export
  mod_ls = list(form, size, cp, plot_performance)
  names(mod_ls) = c('Formula', 'Size', 'Complexity Parameter', 'Performance Plot')
  return(mod_ls)
}


getTree = function(df, form, cp){
  # Create Plot
  mrt = mvpart(form, data = df, pretty = T, xv = "min", minauto = T,
               which = 2, bord = T, uniform = F, text.add = T, branch = 1,
               xadj = .7, yadj = 2, use.n = T, margin = 0.6, keep.y = F,
               bars = F, all.leaves = F, control = rpart.control(cp = cp, xval = 10),
               plot.add = T, prn=F)
  tree_plot = recordPlot()
  #Add column to df that records which leave each portfolio belongs in
  leaf.assign = cbind(as.matrix(mrt$where), df) 
  colnames(leaf.assign) = c("Leaf", colnames(df))
  #specify coordinates at which leaves will be plotted (generates false plots)
  leaves = as.numeric(levels(as.factor(mrt$where)))
  getxy = as.data.frame(cbind(plot(mrt, uniform = F)$x, plot(mrt, uniform = F)$y))
  #add boxplots to leaves
  print(tree_plot)
  plot_adjs_x = c(-0.90, -0.90, -0.70, -0.50, -0.30, -0.15, -0.00)
  plot_adjs_y = c(-0.20, -0.16, -0.16, -0.16, -0.17, -0.16, -0.16)
  for (i in 1:length(leaves)){
    subplot(boxplot.matrix(as.matrix(df[which(leaf.assign$Leaf==leaves[i]), scale_obj_labs]),
                           col = colors, xaxt = 'n', xlab = paste("Leaf",i),
                           cex = .5, labels=F, bg='white', ylim = c(0,1), cex.axis=0.5),
            x = getxy$V1[leaves[i]] + plot_adjs_x[i],
            y = getxy$V2[leaves[i]] + plot_adjs_y[i],
            size = c(0.4, .6))
  }
  legend("topleft", legend = scale_obj_display_labs, fill = colors, bty = "n")
  tree_plot = recordPlot()
  return(tree_plot)
}


# Main
df = init()
mod_ls = getModelHyperparameters(df)
tree_plot = getTree(df, mod_ls[['Formula']], mod_ls[['Complexity Parameter']])

pdf('Multi-Tree Performance.pdf', width = 8, height = 6, pointsize = 9)
print(mod_ls[['Performance Plot']])
dev.off()

pdf('Multi-Tree.pdf', width = 16, height = 8, pointsize = 3)
print(tree_plot)
dev.off()


