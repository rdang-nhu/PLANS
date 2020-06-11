(define-synthax Block
  ([(Block previous_actions perceptions)
     (let ([new_actions (Action_Block previous_actions 5)])
            (let ([new_new_actions (While new_actions perceptions)])
                          (Action_Block new_new_actions 15)))]))