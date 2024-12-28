
#%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%

# Install necessary Packages and Libraries  
install.packages("rpart")
install.packages("rpart.plot")
library(rpart)
library(rpart.plot)
library(ggplot2)

#%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%

# Preparing the data
#Read the CSV file
eBay_data <- read.csv("eBayAuctions.csv")

# dimensions of the entire dataset
dim(eBay_data)

# Check the column names in the data set
names(eBay_data)

# Convert Duration to a factor (categorical variable) 
eBay_data$Duration <- as.factor(eBay_data$Duration)

#%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%
# Part A
#%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%

# Define the function to create training and testing datasets
# If train =True, it returns train otherwise test
create_train_test <- function(data, size = 0.6, train = TRUE) {
  n_row = nrow(data)
  total_row = round(size * n_row)
  train_sample <- 1:total_row
  if (train == TRUE) {
    return(data[train_sample, ])
  } else {
    return(data[-train_sample, ])
  }
}

# Shuffle the data first
set.seed(678)  # Ensures reproducibility
shuffle_index <- sample(1:nrow(eBay_data))
eBay_data <- eBay_data[shuffle_index, ]

# Create the training and validation datasets
train_data <- create_train_test(eBay_data, 0.6, train = TRUE)
validation_data <- create_train_test(eBay_data, 0.6, train = FALSE)

# Check dimensions to confirm the split
cat("Training set dimensions:", dim(train_data), "\n")
cat("Validation set dimensions:", dim(validation_data), "\n")


# Fit the classification tree with `Competitive.` as the target variable
tree_model <- rpart(`Competitive.` ~ ., data = train_data, 
                    method = "class", 
                    control = rpart.control(minbucket = 50, maxdepth = 7, cp = 0))


# Prune the Tree to the Best Size
# To find the best-pruned tree, we can use cross-validation 
# Cross-validation helps to select the optimal complexity parameter


# Print complexity parameter (cp) table to choose the best cp
printcp(tree_model)

# Select the best cp based on the cross-validation error
best_cp <- tree_model$cptable[which.min(tree_model$cptable[,"xerror"]), "CP"]

# Prune the tree using the best cp
pruned_tree <- prune(tree_model, cp = best_cp)

# Display and Interpret the Tree and its Rule
# Plot the pruned tree
rpart.plot(pruned_tree, type = 3, extra = 104, fallen.leaves = TRUE)

# Extract rules from the pruned tree
rules <- path.rpart(pruned_tree, nodes = row.names(pruned_tree$frame))
rules


#%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%

# Simplification by Reducing Less Effective Variables

# Step 1: Remove the 'Category' variable
train_data_no_category <- subset(train_data, select = -c(Category))
validation_data_no_category <- subset(validation_data, select = -c(Category))

# Step 2: Fit the classification tree without 'Category'
tree_model_no_category <- rpart(`Competitive.` ~ ., 
                                data = train_data_no_category, 
                                method = "class", 
                                control = rpart.control(minbucket = 50, maxdepth = 7, cp = 0))

# Step 3: Prune the tree to the best size
# Print complexity parameter (cp) table to choose the best cp
printcp(tree_model_no_category)

# Select the best cp based on the cross-validation error
best_cp_no_category <- tree_model_no_category$cptable[which.min(tree_model_no_category$cptable[,"xerror"]), "CP"]

# Prune the tree using the best cp
pruned_tree_no_category <- prune(tree_model_no_category, cp = best_cp_no_category)

# Step 4: Display and interpret the pruned tree
rpart.plot(pruned_tree_no_category, 
           type = 3,               
           extra = 104,           
           fallen.leaves = TRUE,   
           main = "Decision Tree After Removing 'Category'")

# Step 5: Extract rules from the pruned tree
rules_no_category <- path.rpart(pruned_tree_no_category, nodes = row.names(pruned_tree_no_category$frame))
rules_no_category

#%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%

# Calculate accuracy for the original model (with Category)
pred_original <- predict(pruned_tree, validation_data, type = "class")
conf_matrix_original <- table(Predicted = pred_original, Actual = validation_data$Competitive.)
accuracy_original <- sum(diag(conf_matrix_original)) / sum(conf_matrix_original)

# Calculate accuracy for the reduced model (without Category)
pred_no_category <- predict(pruned_tree_no_category, validation_data_no_category, type = "class")
conf_matrix_no_category <- table(Predicted = pred_no_category, Actual = validation_data_no_category$Competitive.)
accuracy_no_category <- sum(diag(conf_matrix_no_category)) / sum(conf_matrix_no_category)

# Print the comparison of accuracies
cat("Validation Accuracy Comparison:\n")
cat("Original Model (with Category):", accuracy_original, "\n")
cat("Reduced Model (without Category):", accuracy_no_category, "\n")

#%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%

#############################
# Part D                    #
#############################

# Read the dataset and rename the target column
eBay_data <- read.csv("eBayAuctions.csv")

# Rename 'Competitive?' to 'Competitive' 
colnames(eBay_data)[which(names(eBay_data) == "Competitive?")] <- "Competitive"

# Set 'Competitive' as a factor
eBay_data$Competitive <- as.factor(eBay_data$Competitive)

# Select predictors which are available at the start of the auction
actionable_vars <- c("OpenPrice", "Duration", "endDay", "sellerRating")

# Shuffle and split data into training and validation sets
set.seed(678)
shuffle_index <- sample(1:nrow(eBay_data))
eBay_data <- eBay_data[shuffle_index, ]

train_data <- create_train_test(eBay_data, 0.6, train = TRUE)
validation_data <- create_train_test(eBay_data, 0.6, train = FALSE)

tree_model_actionable <- rpart(Competitive ~ ., 
                               data = train_data[, c(actionable_vars, "Competitive")], 
                               method = "class", 
                               control = rpart.control(minbucket = 50, maxdepth = 7, cp = 0))

# Find best complexity parameter (cp) using cross-validation
printcp(tree_model_actionable)
best_cp_actionable <- tree_model_actionable$cptable[which.min(tree_model_actionable$cptable[,"xerror"]), "CP"]

# Prune the tree using the best cp
pruned_tree_actionable <- prune(tree_model_actionable, cp = best_cp_actionable)

# Plot the pruned tree
rpart.plot(pruned_tree_actionable, type = 3, extra = 104, fallen.leaves = TRUE,
           main = "Pruned Tree")

# Extract rules from the pruned tree
rules_actionable <- path.rpart(pruned_tree_actionable, nodes = row.names(pruned_tree_actionable$frame))
cat("Rules from actionable predictors:\n")
print(rules_actionable)

########################
#   Part E             #
########################

# Read the dataset and rename the target column
eBay_data <- read.csv("eBayAuctions.csv")

# Rename 'Competitive?' to 'Competitive' 
colnames(eBay_data)[which(names(eBay_data) == "Competitive?")] <- "Competitive"

# Set 'Competitive' as a factor
eBay_data$Competitive <- as.factor(eBay_data$Competitive)

# Fit a decision tree using 'OpenPrice' and 'Duration' as predictors
tree_model <- rpart(Competitive ~ OpenPrice + Duration, data = eBay_data, method = "class", cp = 0.01)

# Extract splits from the decision tree
splits <- tree_model$splits
split1 <- as.numeric(splits[1, "index"])  
split2 <- as.numeric(splits[2, "index"])

# Create scatter plot 
ggplot(eBay_data, aes(x = OpenPrice, y = Duration, color = Competitive)) +
  geom_point(size = 3) +
  geom_vline(xintercept = split1, linetype = "dashed", color = "green", size = 1) +  # Vertical split on OpenPrice
  geom_hline(yintercept = split2, linetype = "dashed", color = "orange", size = 1) +   # Horizontal split on Duration
  labs(
    title = "Scatter Plot",
    x = "OpenPrice",
    y = "Duration",
    color = "Competitive"
  ) +
  theme_minimal()

# Print the decision tree structure for reference
print(tree_model)

